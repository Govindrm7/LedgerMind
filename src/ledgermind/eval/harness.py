"""Eval harness: score a system's raw completions on a fixed question set.

Predictions are JSONL records ``{"id", "completion", "latency_s", "prompt_tokens",
"completion_tokens"}``. Two scoring modes:

* ``pipeline``: completions follow the LedgerMind schema and go through the verifier.
* ``direct``: free form answers (a frontier model answering directly); the final number
  is extracted and compared, with no verification possible.

Pipeline metrics:

==========================  ==========================================================
verified_accuracy           answered by the verifier AND correct, over all questions.
                            The headline: what a user can trust.
coverage                    share of questions the verifier lets through
selective_accuracy          correct among answered questions
ungated_accuracy            correct if every executed value were returned, verifier or
                            not. The gap to verified_accuracy is what the gate costs.
fabrication_rate            parsed answers citing a figure that is not where it is cited
parse_rate, abstain_rate    format reliability and explicit refusals
==========================  ==========================================================

For counterfactual sets, ``recalled_original_rate`` counts answers that match the gold
answer of the real filing rather than of the edited document.

Usage::

    uv run python -m ledgermind.eval.harness --examples data/processed/finqa/test.jsonl \\
        --predictions outputs/sft_test.jsonl --name sft --out docs/results/eval_sft.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ledgermind.data.convert import ConvertedExample
from ledgermind.data.prepare import from_record
from ledgermind.eval.answer import MatchMode, answers_match, answers_match_stated
from ledgermind.eval.stats import bootstrap_ci, percentile
from ledgermind.numbers import extract_numbers
from ledgermind.schema import to_json
from ledgermind.verifier import verify
from ledgermind.verifier.integrity import check_integrity

_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_FABRICATION = frozenset({"fabricated", "misattributed", "bad_pointer"})


@dataclass
class Row:
    id: str
    status: str
    value: float | bool | None
    correct_strict: bool
    correct_scale: bool
    accepted: bool
    parsed: bool
    abstained: bool
    fabricated: bool
    reasons: list[str]
    recalled_original: bool = False
    latency_s: float | None = None
    completion_tokens: int | None = None


def load_examples(path: Path) -> tuple[list[ConvertedExample], dict[str, float | bool]]:
    """Examples plus, for counterfactual sets, the original filing's gold answers."""
    examples, originals = [], {}
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(from_record(record))
            if "original_gold" in record:
                originals[record["id"]] = record["original_gold"]
    return examples, originals


def load_predictions(path: Path) -> dict[str, dict]:
    with Path(path).open() as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def parse_direct_answer(text: str) -> float | bool | None:
    """Extract the final answer from free text: an ``Answer:`` line, else the last number."""
    lines = _ANSWER_LINE.findall(text)
    tail = lines[-1] if lines else text
    lowered = tail.strip().lower().rstrip(".")
    explicit = bool(lines) or lowered in ("yes", "no")
    if explicit and re.match(r"^(yes|no)\b", lowered):
        return lowered.startswith("yes")
    mentions = extract_numbers(tail)
    if not mentions:
        return None
    m = mentions[0] if lines else mentions[-1]
    return m.value


_FRACTION = re.compile(r"^\$?\s*(-?\d[\d,]*\.?\d*)\s*/\s*(-?\d[\d,]*\.?\d*)\s*%?$")


def parse_direct_answer_stated(text: str) -> tuple[float | bool | None, int | None]:
    """Like ``parse_direct_answer``, plus the number of decimals the answer was written with.

    A final answer written as a simple fraction ("57100/163000") is evaluated, and its
    decimals are None because the value is computed rather than stated.
    """
    lines = _ANSWER_LINE.findall(text)
    tail = lines[-1] if lines else text
    lowered = tail.strip().lower().rstrip(".")
    if (bool(lines) or lowered in ("yes", "no")) and re.match(r"^(yes|no)\b", lowered):
        return lowered.startswith("yes"), None
    fraction = _FRACTION.match(lowered) if lines else None
    if fraction:
        num, den = (float(x.replace(",", "")) for x in fraction.groups())
        return (num / den if den else None), None
    mentions = extract_numbers(tail)
    if not mentions:
        return None, None
    m = mentions[0] if lines else mentions[-1]
    digits = re.sub(r"[^\d.]", "", tail[m.start : m.end])
    return m.value, len(digits.split(".", 1)[1]) if "." in digits else 0


def score_pipeline(
    ex: ConvertedExample, pred: dict | None, original: float | bool | None, issues=None
) -> Row:
    completion = (pred or {}).get("completion") or ""
    verdict = verify(completion, ex.document, integrity_issues=issues)
    value = verdict.value
    accepted = verdict.accepted
    return Row(
        id=ex.id,
        status=verdict.status,
        value=value,
        correct_strict=accepted and answers_match(value, ex.gold_answer, "strict"),
        correct_scale=accepted and answers_match(value, ex.gold_answer, "scale"),
        accepted=accepted,
        parsed=verdict.status != "invalid",
        abstained=verdict.status == "abstained",
        fabricated=bool(verdict.codes & _FABRICATION),
        reasons=sorted(verdict.codes),
        recalled_original=original is not None
        and value is not None
        and answers_match(value, original, "scale")
        and not answers_match(value, ex.gold_answer, "scale"),
        latency_s=(pred or {}).get("latency_s"),
        completion_tokens=(pred or {}).get("completion_tokens"),
    )


def score_direct(
    ex: ConvertedExample,
    pred: dict | None,
    original: float | bool | None,
    stated_precision: bool = False,
) -> Row:
    text = (pred or {}).get("completion") or ""
    if stated_precision:
        value, decimals = parse_direct_answer_stated(text)

        def match(v, gold, mode):
            return answers_match_stated(v, decimals, gold, mode)
    else:
        value, match = parse_direct_answer(text), answers_match
    return Row(
        id=ex.id,
        status="direct",
        value=value,
        correct_strict=match(value, ex.gold_answer, "strict"),
        correct_scale=match(value, ex.gold_answer, "scale"),
        accepted=value is not None,
        parsed=value is not None,
        abstained=False,
        fabricated=False,
        reasons=[],
        recalled_original=original is not None
        and value is not None
        and match(value, original, "scale")
        and not match(value, ex.gold_answer, "scale"),
        latency_s=(pred or {}).get("latency_s"),
        completion_tokens=(pred or {}).get("completion_tokens"),
    )


def _ungated_correct(row: Row, ex: ConvertedExample, mode: MatchMode) -> bool:
    return row.value is not None and answers_match(row.value, ex.gold_answer, mode)


def summarize(rows: list[Row], examples: list[ConvertedExample], counterfactual: bool) -> dict:
    by_id = {e.id: e for e in examples}
    n = len(rows)

    def ci(flags) -> dict:
        return bootstrap_ci([float(f) for f in flags]).as_dict()

    answered = [r for r in rows if r.accepted]
    parsed_answers = [r for r in rows if r.parsed and not r.abstained]
    latencies = [r.latency_s for r in rows if r.latency_s is not None]
    tokens = [r.completion_tokens for r in rows if r.completion_tokens is not None]
    summary = {
        "n": n,
        "verified_accuracy_strict": ci(r.correct_strict for r in rows),
        "verified_accuracy": ci(r.correct_scale for r in rows),
        "ungated_accuracy": ci(_ungated_correct(r, by_id[r.id], "scale") for r in rows),
        "coverage": ci(r.accepted for r in rows),
        "selective_accuracy": ci(r.correct_scale for r in answered) if answered else None,
        "parse_rate": round(sum(r.parsed for r in rows) / n, 4),
        "abstain_rate": round(sum(r.abstained for r in rows) / n, 4),
        "fabrication_rate": (
            round(sum(r.fabricated for r in parsed_answers) / len(parsed_answers), 4)
            if parsed_answers
            else None
        ),
        "rejection_reasons": dict(Counter(c for r in rows for c in r.reasons).most_common()),
    }
    if latencies:
        summary["latency_s"] = {
            "p50": round(percentile(latencies, 0.5), 4),
            "p95": round(percentile(latencies, 0.95), 4),
        }
    if tokens:
        summary["completion_tokens_mean"] = round(sum(tokens) / len(tokens), 1)
    if counterfactual:
        summary["recalled_original_rate"] = ci(r.recalled_original for r in rows)
    return summary


def evaluate(
    examples: list[ConvertedExample],
    predictions: dict[str, dict],
    *,
    direct: bool = False,
    originals: dict[str, float | bool] | None = None,
    stated_precision: bool = False,
) -> tuple[dict, list[Row]]:
    originals = originals or {}
    rows = []
    for ex in examples:
        pred = predictions.get(ex.id)
        original = originals.get(ex.id)
        if direct:
            rows.append(score_direct(ex, pred, original, stated_precision))
        else:
            rows.append(score_pipeline(ex, pred, original, check_integrity(ex.document)))
    return summarize(rows, examples, counterfactual=bool(originals)), rows


def oracle_predictions(examples: list[ConvertedExample]) -> dict[str, dict]:
    """Gold targets as completions. Used to validate the harness end to end."""
    return {ex.id: {"id": ex.id, "completion": to_json(ex.target)} for ex in examples}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Score completions with the verifier")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--oracle", action="store_true", help="score gold targets")
    parser.add_argument("--direct", action="store_true", help="free form answers")
    parser.add_argument(
        "--stated-precision",
        action="store_true",
        help="direct answers match at the precision they state; fractions are evaluated",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=Path, help="optional per question JSONL")
    args = parser.parse_args(argv)

    examples, originals = load_examples(args.examples)
    if args.oracle:
        predictions = oracle_predictions(examples)
    elif args.predictions:
        predictions = load_predictions(args.predictions)
    else:
        parser.error("pass --predictions or --oracle")

    if args.stated_precision and not args.direct:
        parser.error("--stated-precision applies to --direct runs only")
    summary, rows = evaluate(
        examples,
        predictions,
        direct=args.direct,
        originals=originals,
        stated_precision=args.stated_precision,
    )
    summary = {
        "system": args.name,
        "examples": str(args.examples),
        "direct_matching": ("stated_precision" if args.stated_precision else "default")
        if args.direct
        else None,
        **summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.rows:
        with args.rows.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row)) + "\n")
    va, cov = summary["verified_accuracy"], summary["coverage"]
    print(
        f"{args.name}: verified accuracy {va['mean']:.2%} [{va['low']:.2%}, {va['high']:.2%}], "
        f"coverage {cov['mean']:.2%}, fabrication {summary['fabrication_rate']}"
    )
    return summary


if __name__ == "__main__":
    main()
