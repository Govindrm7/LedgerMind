"""Difficulty filtering for GRPO prompts (dynamic sampling in the style of DAPO).

GRPO's advantage is computed relative to the group of completions for one prompt. If all
completions get the same reward, every advantage is zero and the prompt contributes no
gradient. Prompts the SFT model always solves, or never solves, are dead weight. Before
training we sample k completions per prompt from the SFT model and keep only prompts with
a pass rate strictly between 0 and 1.
"""

from __future__ import annotations

from collections.abc import Sequence


def pass_rates(correct: dict[str, Sequence[bool]]) -> dict[str, float]:
    return {pid: sum(flags) / len(flags) for pid, flags in correct.items() if flags}


def informative_prompts(rates: dict[str, float], low: float = 0.0, high: float = 1.0) -> list[str]:
    """Prompt ids whose pass rate lies strictly inside (low, high)."""
    return sorted(pid for pid, rate in rates.items() if low < rate < high)


def summarize_rates(rates: dict[str, float]) -> dict:
    n = len(rates)
    if n == 0:
        return {"prompts": 0}
    solved = sum(r == 1.0 for r in rates.values())
    unsolved = sum(r == 0.0 for r in rates.values())
    return {
        "prompts": n,
        "always_solved": solved,
        "never_solved": unsolved,
        "informative": n - solved - unsolved,
        "mean_pass_rate": round(sum(rates.values()) / n, 4),
    }
