import pytest

from ledgermind.schema import Answer, Evidence, TableSource
from ledgermind.verifier.hygiene import check_hygiene


def answer(plan: str, n: int = 2, unit: str = "number") -> Answer:
    evidence = [
        Evidence(id=f"e{i}", value=10.0 * i, source=TableSource(row=i, col=1))
        for i in range(1, n + 1)
    ]
    return Answer(evidence=evidence, plan=plan, answer_unit=unit)


def codes(plan: str, **kw) -> list[str]:
    return [code for code, _ in check_hygiene(answer(plan, **kw)).errors]


def test_clean_plan():
    result = check_hygiene(answer("(e1 - e2) / e2"))
    assert result.ok and result.unused_evidence == []


@pytest.mark.parametrize(
    "plan, code",
    [
        ("1.64", "bypass"),
        ("e1 * 0 + 42", "bypass"),
        ("e1 - e1 + 12", "bypass"),
        ("e1 / e1 * 100", "bypass"),
        ("100", "bypass"),
        ("e1 + e7", "unknown_ref"),
        ("open(e1)", "plan_invalid"),
        ("(e1 -", "plan_invalid"),
    ],
)
def test_rejections(plan, code):
    assert code in codes(plan)


def test_max_still_depends_on_evidence():
    assert codes("max(e1, e2)") == []


def test_unused_evidence_is_a_warning():
    result = check_hygiene(answer("e1 * 2", n=3))
    assert result.ok
    assert result.unused_evidence == ["e2", "e3"]


def test_unit_must_match_result_type():
    assert codes("greater(e1, e2)", unit="boolean") == []
    assert "unit_mismatch" in codes("greater(e1, e2)", unit="number")
    assert "unit_mismatch" in codes("e1 - e2", unit="boolean")
