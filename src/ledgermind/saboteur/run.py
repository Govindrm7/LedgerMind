"""Run the Saboteur against gold targets and write detection rates.

Usage::

    uv run python -m ledgermind.saboteur.run --split dev
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.saboteur.output_faults import FAULTS
from ledgermind.verifier import verify


def run_output_faults(examples, seed: int = 0) -> dict:
    """Mode A. Returns per fault: cases, detected (any rejection), detected_as_expected."""
    rng = random.Random(seed)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        for name, make in FAULTS.items():
            case = make(ex.target, ex.document, rng)
            if case is None:
                continue
            verdict = verify(case.answer, ex.document, integrity=False)
            s = stats[name]
            s["cases"] += 1
            s["detected"] += not verdict.accepted
            s["detected_as_expected"] += bool(verdict.codes & case.expected)
    return {
        name: {
            **s,
            "detection_rate": round(s["detected"] / s["cases"], 4),
            "expected_code_rate": round(s["detected_as_expected"] / s["cases"], 4),
        }
        for name, s in stats.items()
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Saboteur fault injection")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data", type=Path, default=Path("data/processed/finqa"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/results/saboteur.json"))
    args = parser.parse_args(argv)

    examples = read_jsonl(args.data / f"{args.split}.jsonl")
    results = {"split": args.split, "seed": args.seed, "examples": len(examples)}
    results["mode_a_output_faults"] = run_output_faults(examples, args.seed)
    for name, r in results["mode_a_output_faults"].items():
        print(f"A {name:>20}: {r['detected']}/{r['cases']} caught ({r['detection_rate']:.2%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
