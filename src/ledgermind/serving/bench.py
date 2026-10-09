"""Latency and throughput under a concurrency sweep, against a running server.

Usage::

    uv run python -m ledgermind.serving.bench --base-url http://localhost:8000/v1 \\
        --model ledgermind --examples data/processed/finqa/test.jsonl \\
        --concurrency 1 4 16 64 --requests 256 --out docs/results/bench_grpo_fp16.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.data.prompts import build_prompt
from ledgermind.eval.stats import percentile
from ledgermind.serving.client import ClientConfig, Generation, generate_all


def summarize_level(concurrency: int, results: list[Generation], wall_s: float) -> dict:
    ok = [g for g in results if g.error is None]
    latencies = [g.latency_s for g in ok]
    out_tokens = sum(g.completion_tokens or 0 for g in ok)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": round(wall_s, 3),
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s else None,
        "output_tokens_per_s": round(out_tokens / wall_s, 1) if wall_s else None,
        "latency_p50_s": round(percentile(latencies, 0.5), 4) if latencies else None,
        "latency_p95_s": round(percentile(latencies, 0.95), 4) if latencies else None,
    }


async def sweep(cfg: ClientConfig, items, levels: list[int], transport=None) -> list[dict]:
    rows = []
    for level in levels:
        await generate_all(cfg, items[: min(len(items), level)], level, transport)  # warm up
        results, wall = await generate_all(cfg, items, level, transport)
        rows.append(summarize_level(level, results, wall))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Concurrency sweep benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples)[: args.requests]
    items = [(ex.id, build_prompt(ex.question, ex.document)) for ex in examples]
    cfg = ClientConfig(args.base_url, args.model, constrained=args.constrained)
    rows = asyncio.run(sweep(cfg, items, args.concurrency))
    result = {
        "model": args.model,
        "label": args.label,
        "constrained": args.constrained,
        "levels": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(
            f"c={r['concurrency']:>3}: {r['requests_per_s']} req/s, "
            f"{r['output_tokens_per_s']} tok/s, p50 {r['latency_p50_s']}s, "
            f"p95 {r['latency_p95_s']}s, errors {r['errors']}"
        )


if __name__ == "__main__":
    main()
