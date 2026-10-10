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
import time
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.data.prompts import build_direct_prompt, build_prompt
from ledgermind.serving.client import Budget, ClientConfig, generate_all


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, help="sampling seed sent with every request")
    parser.add_argument(
        "--reasoning", action="store_true", help="OpenAI reasoning model request parameters"
    )
    parser.add_argument("--reasoning-effort", help="reasoning effort; omit for the model default")
    parser.add_argument("--max-cost-usd", type=float, help="hard cap on total spend for paid APIs")
    parser.add_argument("--price-input", type=float, help="USD per 1M input tokens (with a cap)")
    parser.add_argument("--price-output", type=float, help="USD per 1M output tokens (with a cap)")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("outputs/frontier/spend.json"),
        help="spend shared across runs, so one cap covers all of them",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key-env", help="environment variable holding the API key")
    args = parser.parse_args(argv)
    if args.max_cost_usd is not None and (args.price_input is None or args.price_output is None):
        parser.error("--max-cost-usd needs --price-input and --price-output")

    examples = read_jsonl(args.examples)[: args.limit]
    build = build_direct_prompt if args.direct else build_prompt
    items = [(ex.id, build(ex.question, ex.document)) for ex in examples]
    cfg = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        endpoint=args.endpoint,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        max_tokens=args.max_tokens if not args.direct else max(args.max_tokens, 1024),
        temperature=args.temperature,
        seed=args.seed,
        reasoning=args.reasoning,
        reasoning_effort=args.reasoning_effort,
        constrained=args.constrained,
    )
    budget, ledger = None, {"spent_usd": 0.0, "runs": []}
    if args.max_cost_usd is not None:
        if args.ledger.exists():
            ledger = json.loads(args.ledger.read_text())
        budget = Budget(args.max_cost_usd, args.price_input, args.price_output, ledger["spent_usd"])
    results, wall = asyncio.run(generate_all(cfg, items, args.concurrency, budget=budget))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for g in results:
            fh.write(json.dumps(g.as_record()) + "\n")
    errors = sum(g.error is not None for g in results)
    print(f"{len(results)} predictions in {wall:.1f}s ({errors} errors) -> {args.out}")
    if budget is not None:
        run_usd = budget.spent_usd - ledger["spent_usd"]
        ledger["runs"].append(
            {
                "out": str(args.out),
                "model": args.model,
                "sent": budget.sent,
                "skipped": budget.skipped,
                "usd": round(run_usd, 4),
                "usd_per_million_input": args.price_input,
                "usd_per_million_output": args.price_output,
                "finished_at": time.time(),
            }
        )
        ledger["spent_usd"] = budget.spent_usd
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(ledger, indent=2) + "\n")
        print(
            f"spend: this run ${run_usd:.4f}, total ${budget.spent_usd:.4f} of "
            f"${args.max_cost_usd:.2f} cap ({budget.skipped} prompts not sent)"
        )


if __name__ == "__main__":
    main()
