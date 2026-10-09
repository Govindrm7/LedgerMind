"""Sample k completions per prompt from the SFT model and keep informative prompts.

Requires vLLM (GPU host). Writes ``{"summary": ..., "informative": [ids], "rates": ...}``.

Usage::

    python training/pass_rate.py --model checkpoints/sft_qwen3_4b_merged \\
        --train-file data/processed/finqa/train.jsonl --k 8 --out outputs/pass_rates.json
"""

from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams

from ledgermind.data.prepare import read_jsonl
from ledgermind.data.prompts import build_prompt
from ledgermind.rewards import compute_reward
from ledgermind.training.pass_rate import informative_prompts, pass_rates, summarize_rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    examples = read_jsonl(args.train_file)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=0)
    params = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate([build_prompt(e.question, e.document) for e in examples], params)

    correct = {}
    for ex, out in zip(examples, outputs, strict=True):
        correct[ex.id] = [
            compute_reward(c.text, ex.document, ex.gold_answer).terms["correct"] > 0
            for c in out.outputs
        ]
    rates = pass_rates(correct)
    result = {
        "summary": summarize_rates(rates),
        "informative": informative_prompts(rates),
        "rates": rates,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(result["summary"])


if __name__ == "__main__":
    main()
