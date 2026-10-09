"""Oracle check: run converted gold targets through the executor and score them.

This validates the number parser, the converter, the executor and the answer matcher
together. It also gives the accuracy ceiling of the pipeline: an example that cannot be
grounded can never be answered correctly with verified evidence.

Usage::

    uv run python -m ledgermind.eval.oracle --data data/processed/finqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ledgermind.data.prepare import SPLITS, read_jsonl
from ledgermind.dsl import execute
from ledgermind.eval.answer import answers_match


def oracle_split(path: Path, total_raw: int) -> dict:
    examples = read_jsonl(path)
    strict = scale = 0
    mismatches = []
    for ex in examples:
        value = execute(ex.target.plan, {e.id: e.value for e in ex.target.evidence})
        ok = answers_match(value, ex.gold_answer, "strict")
        strict += ok
        scale += answers_match(value, ex.gold_answer, "scale")
        if not ok:
            mismatches.append(
                {"id": ex.id, "plan": ex.target.plan, "value": value, "gold": ex.gold_answer}
            )
    n = len(examples)
    return {
        "grounded": n,
        "total": total_raw,
        "strict_accuracy_on_grounded": round(strict / n, 4),
        "scale_accuracy_on_grounded": round(scale / n, 4),
        "pipeline_ceiling_strict": round(strict / total_raw, 4),
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Oracle check over converted FinQA splits")
    parser.add_argument("--data", type=Path, default=Path("data/processed/finqa"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/finqa_oracle.json"))
    args = parser.parse_args(argv)

    conversion = json.loads((args.data / "report.json").read_text())
    results = {s: oracle_split(args.data / f"{s}.jsonl", conversion[s]["total"]) for s in SPLITS}
    for split, r in results.items():
        print(
            f"{split:>5}: oracle strict {r['strict_accuracy_on_grounded']:.2%} on grounded, "
            f"ceiling {r['pipeline_ceiling_strict']:.2%} of all {r['total']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
