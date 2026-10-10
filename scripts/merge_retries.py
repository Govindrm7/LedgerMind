"""Replace failed requests in a predictions file with the answers from a retry run.

A request that timed out or failed carries an ``error`` and no completion; a retry run
over just those question ids fills them in. Successful originals are never touched, so
the merged file differs from the original only where the original had nothing.

Usage::

    python scripts/merge_retries.py --base run.jsonl --retry run_retry.jsonl --out run_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(base: list[dict], retry: dict[str, dict]) -> tuple[list[dict], int, int]:
    out, filled, still_failed = [], 0, 0
    for row in base:
        if row.get("error") and row["id"] in retry:
            new = retry[row["id"]]
            out.append(new)
            filled += new.get("error") is None
            still_failed += new.get("error") is not None
        else:
            out.append(row)
            still_failed += row.get("error") is not None and row["id"] not in retry
    return out, filled, still_failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    base = [json.loads(line) for line in args.base.read_text().splitlines()]
    retry = {r["id"]: r for r in map(json.loads, args.retry.read_text().splitlines())}
    merged, filled, still_failed = merge(base, retry)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in merged))
    print(f"{args.out}: filled {filled} failed requests, {still_failed} still failed")


if __name__ == "__main__":
    main()
