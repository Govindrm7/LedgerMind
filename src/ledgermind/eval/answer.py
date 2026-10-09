"""Answer matching rules, shared by the oracle check, the GRPO reward and the eval harness.

Two modes, applied identically to every system:

* ``strict``: the predicted value matches the gold value within
  ``max(ABS_TOL, REL_TOL * |gold|)``. FinQA gold answers are rounded to five decimals,
  so ``ABS_TOL`` is 1e-5.
* ``scale``: strict match against the gold value, ``gold * 100`` or ``gold / 100``.
  FinQA often stores percentages as fractions (0.0162 for 1.62%), and this mode stops a
  correct answer in the other convention from being scored as wrong.

Boolean questions match only a boolean prediction with the same value.
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
        return any(_close(pred, g) for g in (gold, gold * 100, gold / 100))
    raise ValueError(f"unknown match mode: {mode}")
