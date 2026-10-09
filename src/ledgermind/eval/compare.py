"""Paired comparison of two systems scored on the same questions.

Reads the per question rows the harness writes (``--rows``) and tests whether system A
differs from system B on verified correctness. Both runs must cover the same question ids,
and both should come from deterministic decoding, or run to run noise leaks into the test.

Usage::

    python -m ledgermind.eval.compare --a outputs/eval/sft_test_rows.jsonl --a-name sft \\
        --b outputs/eval/base_test_rows.jsonl --b-name base --out docs/results/sft_vs_base.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ledgermind.eval.stats import mcnemar, paired_bootstrap

METRICS = ("correct_scale", "correct_strict")


def load_rows(path: Path) -> dict[str, dict]:
    with path.open() as fh:
        return {row["id"]: row for row in map(json.loads, fh)}


def compare(a: dict[str, dict], b: dict[str, dict], metric: str = "correct_scale") -> dict:
    """McNemar and paired bootstrap of A minus B on ``metric``, paired by question id."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if a.keys() != b.keys():
        missing = sorted(a.keys() ^ b.keys())
        raise ValueError(f"{len(missing)} question ids are not in both runs, e.g. {missing[:3]}")
    ids = sorted(a)
    flags_a = [bool(a[i][metric]) for i in ids]
    flags_b = [bool(b[i][metric]) for i in ids]
    return {
        "metric": metric,
        "n": len(ids),
        "accuracy_a": round(sum(flags_a) / len(ids), 4),
        "accuracy_b": round(sum(flags_b) / len(ids), 4),
        "paired_bootstrap": paired_bootstrap(
            [float(x) for x in flags_a], [float(x) for x in flags_b]
        ),
        "mcnemar": mcnemar(flags_a, flags_b),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True, help="rows JSONL of system A")
    parser.add_argument("--b", type=Path, required=True, help="rows JSONL of system B")
    parser.add_argument("--a-name", default="a")
    parser.add_argument("--b-name", default="b")
    parser.add_argument("--metric", choices=METRICS, default="correct_scale")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    result = {
        "a": args.a_name,
        "b": args.b_name,
        **compare(load_rows(args.a), load_rows(args.b), args.metric),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    pb, mc = result["paired_bootstrap"], result["mcnemar"]
    print(
        f"{args.a_name} vs {args.b_name} ({args.metric}, n={result['n']}): "
        f"{result['accuracy_a']:.2%} vs {result['accuracy_b']:.2%}, "
        f"diff {pb['diff']:+.2%} [{pb['low']:+.2%}, {pb['high']:+.2%}], "
        f"McNemar {mc['only_a']} vs {mc['only_b']} discordant, p={mc['p_value']}"
    )
    return result


if __name__ == "__main__":
    main()
