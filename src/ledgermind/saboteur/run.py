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
from ledgermind.saboteur.document_faults import DOCUMENT_FAULTS
from ledgermind.saboteur.output_faults import FAULTS
from ledgermind.verifier import verify
from ledgermind.verifier.integrity import check_footings, footed_cells


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


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.2%}"


def run_document_faults(examples, seed: int = 0) -> dict:
    """Mode B. Stale answer rejection, and integrity detection for faithful readers."""
    rng = random.Random(seed)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        covered = footed_cells(ex.document)
        for name, make in DOCUMENT_FAULTS.items():
            fault = make(ex.target, ex.document, rng)
            if fault is None:
                continue
            s = stats[name]
            s["cases"] += 1
            s["stale_rejected"] += not verify(ex.target, fault.document).accepted
            if not check_footings(ex.document):
                s["clean_docs"] += 1
                s["doc_flagged"] += bool(check_footings(fault.document))
            if fault.faithful is None:
                continue
            detectable = fault.cell in covered
            flagged = any(
                c.startswith("source_footing") for c in verify(fault.faithful, fault.document).codes
            )
            s["faithful_cases"] += 1
            s["faithful_flagged"] += flagged
            s["detectable"] += detectable
            s["detectable_flagged"] += flagged and detectable
    return {
        name: {
            **s,
            "stale_rejection_rate": _rate(s["stale_rejected"], s["cases"]),
            "document_flag_rate": _rate(s["doc_flagged"], s["clean_docs"]),
            "faithful_detection_rate": _rate(s["faithful_flagged"], s["faithful_cases"]),
            "detectable_share": _rate(s["detectable"], s["faithful_cases"]),
            "detection_rate_when_detectable": _rate(s["detectable_flagged"], s["detectable"]),
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

    results["mode_b_document_faults"] = run_document_faults(examples, args.seed)
    for name, r in results["mode_b_document_faults"].items():
        print(
            f"B {name:>20}: stale rejected {_pct(r['stale_rejection_rate'])}, "
            f"document flagged {_pct(r['document_flag_rate'])}, "
            f"faithful flagged {_pct(r['faithful_detection_rate'])}, "
            f"when detectable {_pct(r['detection_rate_when_detectable'])} "
            f"(detectable share {_pct(r['detectable_share'])})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
