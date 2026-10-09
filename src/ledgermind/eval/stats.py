"""Statistics for honest comparisons.

With about 1,100 FinQA test questions, a 95% interval on accuracy is roughly plus or
minus 2.5 points, so every reported metric carries a bootstrap interval and every
"system A beats system B" claim is backed by a paired test on the same questions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "low": round(self.low, 4), "high": round(self.high, 4)}


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean."""
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return Interval(
        sum(values) / n, _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)
    )


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Difference in means (a minus b) on paired items, with interval and two sided p."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    observed = sum(diffs) / n
    # p value: how often the bootstrap distribution, centered at zero, is as extreme
    extreme = sum(abs(d - observed) >= abs(observed) for d in boots)
    return {
        "diff": round(observed, 4),
        "low": round(_percentile(boots, 0.025), 4),
        "high": round(_percentile(boots, 0.975), 4),
        "p_value": round((extreme + 1) / (n_boot + 1), 4),
    }


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correctness. Counts discordant pairs only."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"only_a": only_a, "only_b": only_b, "p_value": round(min(1.0, 2 * tail), 6)}


def percentile(values: Sequence[float], q: float) -> float:
    return _percentile(sorted(values), q)
