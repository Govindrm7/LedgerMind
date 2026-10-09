import json

import pytest

from ledgermind.document import Document
from ledgermind.verifier.verifier import verify

DOC = Document.build(
    "T",
    table=[
        ["( in millions )", "2020", "2019"],
        ["product revenue", "$ 1200", "$ 1100"],
        ["service revenue", "300", "250"],
        ["total revenue", "$ 1500", "$ 1350"],
        ["margin", "12%", "10%"],
    ],
    sentences=["the company employed 4200 people ."],
)


def output(plan: str, *evidence: tuple, unit: str = "number", **extra) -> str:
    ev = [
        {"id": f"e{i}", "value": v, "label": "", "source": {"type": "table", "row": r, "col": c}}
        for i, (v, r, c) in enumerate(evidence, 1)
    ]
    return json.dumps({"evidence": ev, "plan": plan, "answer_unit": unit, **extra})


GOOD = output("(e1 - e2) / e2", (1500, 3, 1), (1350, 3, 2), unit="ratio")


def test_accepts_grounded_answer_with_value_and_audit():
    verdict = verify(GOOD, DOC)
    assert verdict.accepted
    assert verdict.value == pytest.approx(150 / 1350)
    audit = verdict.audit()
    json.dumps(audit)
    assert audit["status"] == "accepted"
    assert [e["check"] for e in audit["evidence"]] == ["ok", "ok"]
    assert audit["evidence"][0]["cited_text"] == "$ 1500"


def test_invalid_output():
    verdict = verify("the answer is 11.1%", DOC)
    assert verdict.status == "invalid" and verdict.codes == {"schema_invalid"}


def test_abstain():
    verdict = verify('{"abstain": true, "reason": "not reported"}', DOC)
    assert verdict.status == "abstained"
    assert verdict.audit()["abstain_reason"] == "not reported"


@pytest.mark.parametrize(
    "text, code",
    [
        (output("e1 - e2", (1501, 3, 1), (1350, 3, 2)), "fabricated"),
        (output("e1 - e2", (1350, 3, 1), (1350, 3, 2)), "misattributed"),
        (output("e1 - e2", (1500, 9, 1), (1350, 3, 2)), "bad_pointer"),
        (output("e1 - e1 + 12", (1500, 3, 1)), "bypass"),
        (output("e1 * 0.1111", (1500, 3, 1)), "bypass"),
        (output("e1 - e3", (1500, 3, 1), (1350, 3, 2)), "unknown_ref"),
        (output("e1 / (e2 - e2)", (1500, 3, 1), (1350, 3, 2)), "execution_error"),
        (output("e1 - e2", (1500, 3, 1), (12, 4, 1)), "unit_mismatch"),
        (output("greater(e1, e2)", (1500, 3, 1), (1350, 3, 2)), "unit_mismatch"),
    ],
)
def test_rejections(text, code):
    verdict = verify(text, DOC)
    assert verdict.status == "rejected"
    assert code in verdict.codes


def test_all_failures_are_recorded_not_just_the_first():
    verdict = verify(output("e1 - e2", (1501, 3, 1), (12, 4, 1)), DOC)
    assert {"fabricated"} <= verdict.codes
    assert verdict.value == pytest.approx(1501 - 12)


def test_unused_evidence_is_a_warning():
    verdict = verify(output("e1 * 2", (1500, 3, 1), (1350, 3, 2)), DOC)
    assert verdict.accepted
    assert [w.code for w in verdict.warnings] == ["unused_evidence"]
    assert verdict.audit()["evidence"][1]["used"] is False


def test_broken_footing_rejects_answers_that_cite_it():
    tampered = DOC.with_cell(2, 1, "$ 350")
    verdict = verify(GOOD, tampered)
    assert "source_footing_broken" in verdict.codes
    assert verify(GOOD, tampered, integrity=False).accepted


def test_broken_footing_elsewhere_does_not_reject():
    tampered = DOC.with_cell(2, 2, "$ 300")
    text = output("e1 / e2", (1500, 3, 1), (300, 2, 1))
    assert verify(text, tampered).accepted
