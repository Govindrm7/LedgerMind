"""Saboteur Mode A: corrupt model outputs and measure what the verifier catches.

Each fault reproduces a failure mode LLMs actually exhibit on financial documents. The
document is left untouched, so every fault is detectable in principle; the detection
rate is a direct measure of the fabrication guarantee.

========================  ==========================================  ==================
fault                     what changes                                expected catch
========================  ==========================================  ==================
digit_change              one digit of one cited value                fabricated
sign_flip                 sign of one cited value                     fabricated
wrong_column              pointer moved to another column, same row   misattributed
wrong_row                 pointer moved to another row, same column   misattributed
scale_error               a cited value multiplied by 1000            fabricated
percent_as_fraction       a cited percent value divided by 100        fabricated
hardcoded_answer          plan replaced by the final number           bypass
disguised_constant        plan replaced by ``e1 - e1 + k``            bypass
========================  ==========================================  ==================
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ledgermind.document import Document
from ledgermind.dsl import ALLOWED_LITERALS, execute
from ledgermind.numbers import numbers_equal
from ledgermind.schema import Answer, Evidence, TableSource
from ledgermind.verifier.provenance import check_evidence


@dataclass(frozen=True)
class FaultCase:
    fault: str
    answer: Answer
    expected: frozenset[str]
    evidence_id: str | None = None


def _replace_evidence(answer: Answer, new: Evidence) -> Answer:
    evidence = [new if e.id == new.id else e for e in answer.evidence]
    return answer.model_copy(update={"evidence": evidence})


def _printed_at(doc: Document, ev: Evidence, value: float) -> bool:
    return any(numbers_equal(m.value, value) for m in doc.numbers_at(ev.source))


def _with_value(answer: Answer, doc: Document, ev: Evidence, value: float) -> Answer | None:
    """Swap in a corrupted value, unless it happens to be printed at the cited location too."""
    if _printed_at(doc, ev, value):
        return None
    return _replace_evidence(answer, ev.model_copy(update={"value": value}))


def _change_digit(value: float, rng: random.Random) -> float | None:
    text = f"{abs(value):.10f}".rstrip("0").rstrip(".")
    positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    if not positions:
        return None
    i = rng.choice(positions)
    choices = [d for d in "0123456789" if d != text[i] and not (i == 0 and d == "0")]
    mutated = text[:i] + rng.choice(choices) + text[i + 1 :]
    return float(mutated) * (-1 if value < 0 else 1)


def digit_change(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    ev = rng.choice(answer.evidence)
    value = _change_digit(ev.value, rng)
    if value is None:
        return None
    mutated = _with_value(answer, doc, ev, value)
    if mutated is None:
        return None
    expected = frozenset({"fabricated", "misattributed"})
    return FaultCase("digit_change", mutated, expected, ev.id)


def sign_flip(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    candidates = [e for e in answer.evidence if e.value != 0]
    if not candidates:
        return None
    ev = rng.choice(candidates)
    mutated = _with_value(answer, doc, ev, -ev.value)
    if mutated is None:
        return None
    return FaultCase("sign_flip", mutated, frozenset({"fabricated", "misattributed"}), ev.id)


def scale_error(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    ev = rng.choice(answer.evidence)
    mutated = _with_value(answer, doc, ev, ev.value * 1000)
    if mutated is None:
        return None
    return FaultCase("scale_error", mutated, frozenset({"fabricated", "misattributed"}), ev.id)


def percent_as_fraction(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    percents = []
    for e in answer.evidence:
        check = check_evidence(e, doc)
        if check.mention is not None and check.mention.is_percent:
            percents.append(e)
    if not percents:
        return None
    ev = rng.choice(percents)
    mutated = _with_value(answer, doc, ev, ev.value / 100)
    if mutated is None:
        return None
    expected = frozenset({"fabricated", "misattributed"})
    return FaultCase("percent_as_fraction", mutated, expected, ev.id)


def _move_pointer(answer: Answer, doc: Document, rng: random.Random, axis: str) -> FaultCase | None:
    table_ev = [e for e in answer.evidence if isinstance(e.source, TableSource)]
    if not table_ev:
        return None
    ev = rng.choice(table_ev)
    src = ev.source
    if axis == "col":
        options = [TableSource(row=src.row, col=c) for c in range(1, len(doc.table[src.row]))]
    else:
        options = [TableSource(row=r, col=src.col) for r in range(1, doc.n_rows)]
    options = [
        o
        for o in options
        if o != src
        and doc.numbers_at(o)
        and not any(numbers_equal(m.value, ev.value) for m in doc.numbers_at(o))
    ]
    if not options:
        return None
    mutated = _replace_evidence(answer, ev.model_copy(update={"source": rng.choice(options)}))
    fault = "wrong_column" if axis == "col" else "wrong_row"
    return FaultCase(fault, mutated, frozenset({"misattributed", "fabricated"}), ev.id)


def wrong_column(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    return _move_pointer(answer, doc, rng, "col")


def wrong_row(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    return _move_pointer(answer, doc, rng, "row")


def hardcoded_answer(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    value = execute(answer.plan, {e.id: e.value for e in answer.evidence})
    if isinstance(value, bool):
        return None
    literal = f"{value:.6g}"
    if float(literal) in ALLOWED_LITERALS:
        return None
    return FaultCase(
        "hardcoded_answer", answer.model_copy(update={"plan": literal}), frozenset({"bypass"})
    )


def disguised_constant(answer: Answer, doc: Document, rng: random.Random) -> FaultCase | None:
    if answer.answer_unit == "boolean":
        return None
    ev = answer.evidence[0]
    k = rng.choice(sorted(ALLOWED_LITERALS))
    plan = f"{ev.id} - {ev.id} + {k:g}"
    return FaultCase(
        "disguised_constant", answer.model_copy(update={"plan": plan}), frozenset({"bypass"})
    )


FAULTS: dict[str, Callable[[Answer, Document, random.Random], FaultCase | None]] = {
    "digit_change": digit_change,
    "sign_flip": sign_flip,
    "wrong_column": wrong_column,
    "wrong_row": wrong_row,
    "scale_error": scale_error,
    "percent_as_fraction": percent_as_fraction,
    "hardcoded_answer": hardcoded_answer,
    "disguised_constant": disguised_constant,
}
