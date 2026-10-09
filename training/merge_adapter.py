"""Merge a LoRA adapter into its base model, for GRPO initialization and for serving.

Usage::

    uv run python training/merge_adapter.py --base Qwen/Qwen3-4B-Base \\
        --adapter checkpoints/sft_qwen3_4b --out checkpoints/sft_qwen3_4b_merged
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=getattr(torch, args.dtype))
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
