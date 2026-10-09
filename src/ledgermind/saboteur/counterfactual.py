"""Saboteur Mode C: counterfactual documents.

FinQA is built from public 10-K filings that frontier models have very likely seen in
pretraining. Mode C edits the figures a question depends on, so that the correct answer
for the edited document differs from the answer in the real filing. A system that reads
the document gets the counterfactual answer; a system that recalls the filing gets the
original one, and LedgerMind's provenance gate rejects it because the recalled figures
are no longer printed where they are cited.

Every cited figure is scaled by a random factor in [0.90, 0.97] or [1.03, 1.10], printed
with its original precision, sign and format. All other places that print the same figure
are rewritten too, so the edited document stays self consistent and a model cannot fall
back on an untouched copy of the original number. The gold answer is recomputed by running
the gold plan on the new figures.

Usage::

    uv run python -m ledgermind.saboteur.counterfactual --split test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ledgermind.data.convert import ConvertedExample
from ledgermind.data.prepare import read_jsonl, to_record
from ledgermind.document import Document
from ledgermind.dsl import PlanError, execute
from ledgermind.eval.answer import answers_match
from ledgermind.numbers import extract_numbers, numbers_equal
from ledgermind.saboteur.rewrite import decimals_of, format_magnitude, replace_mention
from ledgermind.schema import TableSource
from ledgermind.verifier import verify
from ledgermind.verifier.provenance import check_evidence


@dataclass(frozen=True)
class CounterfactualExample:
    example: ConvertedExample
    original_gold: float | bool
    edits: tuple[tuple[float, float], ...]

    def to_record(self) -> dict:
        record = to_record(self.example)
        record["original_gold"] = self.original_gold
        record["edits"] = [list(e) for e in self.edits]
        return record


def _factor(rng: random.Random) -> float:
    return rng.uniform(0.90, 0.97) if rng.random() < 0.5 else rng.uniform(1.03, 1.10)


def _rewrite_text(text: str, mapping: dict[float, str]) -> str:
    mentions = extract_numbers(text)
    for m in sorted(mentions, key=lambda m: m.start, reverse=True):
        for old, new_magnitude in mapping.items():
            if numbers_equal(m.value, old):
                text = replace_mention(text, m, new_magnitude)
                break
    return text


def rewrite_document(doc: Document, mapping: dict[float, str]) -> Document:
    """Rewrite every printed occurrence of each signed value in ``mapping``."""
    table = [[_rewrite_text(cell, mapping) for cell in row] for row in doc.table]
    sentences = [_rewrite_text(s, mapping) for s in doc.sentences]
    return Document.build(doc.doc_id, table, sentences)


def build_counterfactual(ex: ConvertedExample, rng: random.Random) -> CounterfactualExample | None:
    mapping: dict[float, str] = {}
    new_values: dict[float, float] = {}
    for ev in ex.target.evidence:
        if ev.value in new_values:
            continue
        check = check_evidence(ev, ex.document)
        if not check.ok or check.mention is None or ev.value == 0:
            return None
        if isinstance(ev.source, TableSource) and ev.source.row == 0:
            return None  # header cells (years, dates) are labels, not figures
        decimals = decimals_of(check.cited_text, check.mention)
        for _ in range(10):
            magnitude = format_magnitude(abs(ev.value) * _factor(rng), decimals)
            if float(magnitude) != abs(ev.value) and float(magnitude) != 0:
                break
        else:
            return None
        mapping[ev.value] = magnitude
        new_values[ev.value] = float(magnitude) * (-1 if ev.value < 0 else 1)

    doc = rewrite_document(ex.document, mapping)
    evidence = [e.model_copy(update={"value": new_values[e.value]}) for e in ex.target.evidence]
    target = ex.target.model_copy(update={"evidence": evidence})
    try:
        gold = execute(target.plan, {e.id: e.value for e in evidence})
    except PlanError:
        return None
    if answers_match(gold, ex.gold_answer, "scale"):
        return None  # the edit did not change the answer, so it cannot expose recall
    if not verify(target, doc, integrity=False).accepted:
        return None
    edits = tuple((old, new_values[old]) for old in mapping)
    cf = ConvertedExample(ex.id, ex.question, doc, target, gold, ex.flags)
    return CounterfactualExample(cf, ex.gold_answer, edits)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Build counterfactual FinQA documents")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", type=Path, default=Path("data/processed/finqa"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    examples = read_jsonl(args.data / f"{args.split}.jsonl")
    built = [cf for ex in examples if (cf := build_counterfactual(ex, rng)) is not None]
    out = args.data / f"counterfactual_{args.split}.jsonl"
    with out.open("w") as fh:
        for cf in built:
            fh.write(json.dumps(cf.to_record(), ensure_ascii=False) + "\n")
    stats = {"split": args.split, "source": len(examples), "built": len(built), "seed": args.seed}
    print(f"built {len(built)}/{len(examples)} counterfactual examples -> {out}")
    report = Path("docs/results") / f"counterfactual_{args.split}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


if __name__ == "__main__":
    main()
