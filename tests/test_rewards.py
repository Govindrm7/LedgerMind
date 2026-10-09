import json

import pytest

from ledgermind.document import Document
from ledgermind.rewards import compute_reward

DOC = Document.build(
    "T",
    table=[["", "2020", "2019"], ["revenue", "$ 5829", "$ 5735"], ["margin", "12%", "10%"]],
    sentences=[],
)
GOLD = (5829 - 5735) / 5735


def out(plan, *evidence, unit="ratio", **extra):
    ev = [
        {"id": f"e{i}", "value": v, "label": "", "source": {"type": "table", "row": r, "col": c}}
        for i, (v, r, c) in enumerate(evidence, 1)
    ]
    return json.dumps({"evidence": ev, "plan": plan, "answer_unit": unit, **extra})


CORRECT = out("(e1 - e2) / e2", (5829, 1, 1), (5735, 1, 2))
WRONG_GROUNDED = out("e1 / e2", (5829, 1, 1), (5735, 1, 2))
ABSTAIN = '{"abstain": true, "reason": "unsure"}'
GARBAGE = "the answer is 1.64%"
FABRICATED = out("(e1 - e2) / e2", (5829, 1, 1), (5700, 1, 2))
BYPASS = out("e1 - e1 + 2", (5829, 1, 1), unit="number")


def total(text):
    return compute_reward(text, DOC, GOLD).total


def test_terms_for_a_correct_answer():
    r = compute_reward(CORRECT, DOC, GOLD)
    assert r.terms["correct"] == 1.0 and r.terms["provenance"] == 0.3 and r.terms["schema"] == 0.1
    assert r.total == pytest.approx(1.4)
    assert r.verdict.accepted


def test_outcome_ordering():
    ranked = [CORRECT, WRONG_GROUNDED, ABSTAIN, GARBAGE, FABRICATED, BYPASS]
    totals = [total(t) for t in ranked]
    assert totals == sorted(totals, reverse=True)
    assert len(set(totals)) == len(totals)


def test_fabricated_figures_cannot_earn_correctness():
    # Writing the right numbers at the wrong place yields the right value but no credit.
    misplaced = out("(e1 - e2) / e2", (5829, 1, 2), (5735, 1, 1))
    r = compute_reward(misplaced, DOC, GOLD)
    assert r.terms["correct"] == 0.0 and r.terms["fabrication"] == -0.5


def test_padding_with_real_but_unused_figures_does_not_pay():
    padded = out("(e1 - e2) / e2", (5829, 1, 1), (5735, 1, 2), (12, 2, 1), (10, 2, 2))
    assert total(padded) < total(CORRECT)


def test_smuggled_answer_field_is_a_bypass():
    smuggled = out("(e1 - e2) / e2", (5829, 1, 1), (5735, 1, 2), answer=0.0164)
    r = compute_reward(smuggled, DOC, GOLD)
    assert r.terms["bypass"] == -1.0 and r.terms["schema"] == 0.0


def test_unused_penalty_is_capped():
    many = out("e1 * 2", *[(5829, 1, 1)] + [(5735, 1, 2)] * 1 + [(12, 2, 1), (10, 2, 2)] * 3)
    r = compute_reward(many, DOC, GOLD)
    assert r.terms["unused_evidence"] >= -0.25


def test_percent_convention_is_tolerated_in_scale_mode():
    as_percent = out("(e1 - e2) / e2 * 100", (5829, 1, 1), (5735, 1, 2), unit="percent")
    assert compute_reward(as_percent, DOC, GOLD).terms["correct"] == 1.0
    strict = compute_reward(as_percent, DOC, GOLD, match_mode="strict")
    assert strict.terms["correct"] == 0.0


def test_invariant_violation_costs_reward():
    mixed = out("e1 - e2", (5829, 1, 1), (12, 2, 1), unit="number")
    assert compute_reward(mixed, DOC, 5817).terms["invariant"] == -0.2
