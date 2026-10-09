"""Generate predictions for an eval set from an OpenAI compatible server.

Usage (vLLM serving the fine tuned model)::

    uv run python -m ledgermind.serving.predict --examples data/processed/finqa/test.jsonl \\
        --base-url http://localhost:8000/v1 --model ledgermind --out outputs/pred_grpo.jsonl

Use ``--endpoint chat --direct`` for a frontier baseline that answers in free text, or
``--endpoint chat`` alone to run the frontier model through the LedgerMind pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.data.prompts import build_direct_prompt, build_prompt
from ledgermind.serving.client import ClientConfig, generate_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate predictions from a server")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    parser.add_argument("--direct", action="store_true", help="free form answer baseline")
    parser.add_argument("--constrained", action="store_true", help="JSON schema decoding")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key-env", help="environment variable holding the API key")
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples)[: args.limit]
    build = build_direct_prompt if args.direct else build_prompt
    items = [(ex.id, build(ex.question, ex.document)) for ex in examples]
    cfg = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        endpoint=args.endpoint,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        max_tokens=args.max_tokens if not args.direct else max(args.max_tokens, 1024),
        constrained=args.constrained,
    )
    results, wall = asyncio.run(generate_all(cfg, items, args.concurrency))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for g in results:
            fh.write(json.dumps(g.as_record()) + "\n")
    errors = sum(g.error is not None for g in results)
    print(f"{len(results)} predictions in {wall:.1f}s ({errors} errors) -> {args.out}")


if __name__ == "__main__":
    main()
