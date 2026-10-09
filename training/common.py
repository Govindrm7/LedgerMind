"""Shared plumbing for the training scripts: config loading, run metadata, tokenizer."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_tokenizer(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def fits(tokenizer, prompt: str, completion: str, max_length: int) -> bool:
    n = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
    return n + 1 <= max_length  # +1 for the EOS token appended to the completion


def write_run_metadata(output_dir: str, config: dict, **extra) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "git_sha": git_sha(), "finished_at": time.time(), **extra}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


class _TorchChunkedLogProb:
    """Pure PyTorch stand in for TRL's Triton log prob kernel.

    TRL 1.x routes the training loss through a Triton kernel, and Triton ships with PyTorch
    on Linux only. This fallback computes the same quantities with plain autograd so that
    the CPU smoke tests on macOS run the real trainer code path. GPU runs never use it.
    """

    @staticmethod
    def apply(
        hidden,
        weight,
        bias,
        targets,
        temperature,
        chunk_size,
        softcap,
        logit_scale,
        outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
    ):
        import torch

        logits = hidden.float() @ weight.float().T
        if bias is not None:
            logits = logits + bias.float()
        logits = logits * logit_scale
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        logits = logits / temperature
        log_probs = logits.log_softmax(-1)
        token = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1) if "entropy" in outputs else None
        with torch.no_grad():
            sq = torch.logsumexp(2 * log_probs, -1) if "log_sum_sq_probs" in outputs else None
            mean = logits.mean(-1) if "mean_logits" in outputs else None
            top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return token, entropy, sq, mean, top1


def install_cpu_kernel_fallback() -> bool:
    """Install the fallback if TRL's Triton kernel is unavailable. Returns True if installed."""
    import trl.trainer.utils as trl_utils

    if trl_utils._ChunkedLogProbFunction is not None:
        return False
    trl_utils._ChunkedLogProbFunction = _TorchChunkedLogProb
    return True
