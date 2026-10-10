"""Draw a fixed random subset of a split, for baselines too costly to run on all of it.

The subset is a seeded uniform sample, so a system scored on it can still be compared with
any other system on exactly the same questions. The chosen ids are written next to the
data so the sample can be checked and rebuilt.

Usage::

    python -m ledgermind.data.sample --split test --n 400 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sample_ids(ids: list[str], n: int, seed: int) -> list[str]:
    """``n`` ids drawn uniformly without replacement, in their original order."""
    if n > len(ids):
        raise ValueError(f"cannot sample {n} of {len(ids)}")
    chosen = set(random.Random(seed).sample(ids, n))
    return [i for i in ids if i in chosen]


def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", type=Path, default=Path("data/processed/finqa"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids-out", type=Path, help="also write the chosen ids here")
    args = parser.parse_args(argv)

    lines = (args.data / f"{args.split}.jsonl").read_text().splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    keep = set(sample_ids(ids, args.n, args.seed))
    name = f"{args.split}_sample{args.n}"
    out = args.data / f"{name}.jsonl"
    out.write_text("".join(f"{line}\n" for line, i in zip(lines, ids, strict=True) if i in keep))
    chosen = [i for i in ids if i in keep]
    record = {"split": args.split, "n": args.n, "seed": args.seed, "ids": chosen}
    for path in [args.data / f"{name}_ids.json", args.ids_out]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"sampled {len(chosen)} of {len(ids)} {args.split} questions (seed {args.seed}) -> {out}")
    return chosen


if __name__ == "__main__":
    main()
