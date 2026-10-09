import pytest

from ledgermind.document import Document
from ledgermind.dsl import compile_plan
from ledgermind.schema import Answer, Evidence, TableSource, TextSource
from ledgermind.verifier.invariants import check_invariants
from ledgermind.verifier.provenance import check_provenance

DOC = Document.build(
    "T",
    table=[
        ["( in millions )", "year ended 2020", "q4 2020", "2019"],
        ["revenue", "$ 5829", "$ 1500", "$ 5735"],
        ["margin", "12%", "11%", "10%"],
        ["backlog ( in billions )", "$ 2.1", "$ 2.0", "$ 1.9"],
    ],
    sentences=["the company had 1200 employees and $ 3.8 million of cash ."],
)

REFS = {
    "rev20": (5829, TableSource(row=1, col=1)),
    "rev_q4": (1500, TableSource(row=1, col=2)),
    "rev19": (5735, TableSource(row=1, col=3)),
    "margin20": (12, TableSource(row=2, col=1)),
    "margin19": (10, TableSource(row=2, col=3)),
    "backlog": (2.1, TableSource(row=3, col=1)),
    "cash": (3.8, TextSource(sent=0)),
    "staff": (1200, TextSource(sent=0)),
}


def violations(plan_template: str, *names: str) -> list[str]:
    evidence = [
        Evidence(id=f"e{i}", value=REFS[n][0], source=REFS[n][1]) for i, n in enumerate(names, 1)
    ]
    answer = Answer(evidence=evidence, plan=plan_template, answer_unit="number")
    checks = check_provenance(answer, DOC)
    assert all(c.ok for c in checks)
    return [v.code for v in check_invariants(answer, compile_plan(answer.plan), checks, DOC)]


@pytest.mark.parametrize(
    "plan, names",
    [
        ("(e1 - e2) / e2", ("rev20", "rev19")),
        ("e1 - e2", ("margin20", "margin19")),
        ("e1 * (e2 / 100)", ("rev20", "margin20")),
        ("e1 / 1000 + e2", ("rev20", "backlog")),
        ("e1 / e2", ("rev20", "staff")),
        ("mean(e1, e2)", ("rev20", "rev19")),
    ],
)
def test_compatible_plans_pass(plan, names):
    assert violations(plan, *names) == []


@pytest.mark.parametrize(
    "plan, names, code",
    [
        ("e1 - e2", ("rev20", "margin20"), "unit_mismatch"),
        ("sum(e1, e2)", ("margin20", "rev19"), "unit_mismatch"),
        ("e1 + e2", ("rev20", "backlog"), "scale_mismatch"),
        ("e1 + e2", ("rev20", "rev_q4"), "period_mismatch"),
        ("pct_change(e1, e2)", ("rev_q4", "rev20"), "period_mismatch"),
    ],
)
def test_mismatches_are_flagged(plan, names, code):
    assert code in violations(plan, *names)


def test_table_wide_scale_comes_from_the_header_corner():
    assert violations("e1 + e2", "rev19", "backlog") == ["scale_mismatch"]


def test_unknown_dimensions_are_compatible():
    assert violations("e1 + e2", "cash", "staff") == []
