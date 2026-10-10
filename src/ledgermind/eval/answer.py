"""Answer matching rules, shared by the oracle check, the GRPO reward and the eval harness.

Two modes, applied identically to every system:

* ``strict``: the predicted value matches the gold value within
  ``max(ABS_TOL, REL_TOL * |gold|)``. FinQA gold answers are rounded to five decimals,
  so ``ABS_TOL`` is 1e-5.
* ``scale``: strict match against the gold value with the prediction read as is, divided
  by 100 or multiplied by 100. FinQA often stores percentages as fractions (0.0162 for
  1.62%), and this mode stops a correct answer in the other convention from being scored
  as wrong. The comparison happens in the gold value's own units, so the tolerance tracks
  the gold's five decimal rounding (1.71447 matches 0.01714, whose rounding error grows to
  5e-4 when multiplied by 100).

Boolean questions match only a boolean prediction with the same value.

Direct (free text) baselines can also be scored at the precision they state
(``answers_match_stated``): "1.64" counts as a match for 1.63657 because that is the gold
value rounded to the two decimals given. Pipeline answers are computed exactly by the
executor, so the rule only matters for answers a model writes out itself.
"""

from __future__ import annotations

from typing import Literal

ABS_TOL = 1e-5
REL_TOL = 1e-4

MatchMode = Literal["strict", "scale"]
AnswerValue = float | bool


def _close(pred: float, gold: float) -> bool:
    return abs(pred - gold) <= max(ABS_TOL, REL_TOL * abs(gold))


def answers_match(
    pred: float | bool | None, gold: float | bool, mode: MatchMode = "strict"
) -> bool:
    if pred is None:
        return False
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and pred is gold
    if mode == "strict":
        return _close(pred, gold)
    if mode == "scale":
        return any(_close(p, gold) for p in (pred, pred / 100, pred * 100))
    raise ValueError(f"unknown match mode: {mode}")


def answers_match_stated(
    pred: float | bool | None,
    decimals: int | None,
    gold: float | bool,
    mode: MatchMode = "strict",
) -> bool:
    """Match at the stated precision: ``pred`` is the gold value rounded to ``decimals``.

    The stated number must carry at least two significant digits, so a bare "7" cannot
    match 7.16. ``decimals`` is None for a value that was computed rather than written out
    (a fraction such as 57100/163000), which then needs an ordinary match.
    """
    if answers_match(pred, gold, mode):
        return True
    if decimals is None or pred is None or isinstance(pred, bool) or isinstance(gold, bool):
        return False
    if len(str(abs(round(pred * 10**decimals))).lstrip("0")) < 2:
        return False
    half_unit = 0.5 * 10.0**-decimals
    factors = (1.0,) if mode == "strict" else (1.0, 100.0, 0.01)
    # pred is compared with gold * factor; the gold's own rounding scales with the factor
    return any(abs(pred - gold * f) <= half_unit * (1 + 1e-9) + ABS_TOL * f for f in factors)
