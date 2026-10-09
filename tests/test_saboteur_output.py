import random

import pytest

from ledgermind.document import Document
from ledgermind.saboteur.output_faults import FAULTS
from ledgermind.saboteur.run import run_output_faults
from ledgermind.schema import Answer, Evidence, TableSource
from ledgermind.verifier import verify

DOC = Document.build(
    "T",
    table=[
        ["", "2020", "2019"],
        ["revenue", "$ 5829", "$ 5735"],
        ["margin", "12%", "10%"],
        ["costs", "$ 4100", "$ 3900"],
    ],
    sentences=[],
)
ANSWER = Answer(
    evidence=[
        Evidence(id="e1", value=5829, source=TableSource(row=1, col=1)),
        Evidence(id="e2", value=12, source=TableSource(row=2, col=1)),
    ],
    plan="e1 * (e2 / 100)",
    answer_unit="number",
)


def test_clean_answer_is_accepted():
    assert verify(ANSWER, DOC).accepted


@pytest.mark.parametrize("name", sorted(FAULTS))
def test_every_fault_is_caught_with_its_expected_code(name):
    case = FAULTS[name](ANSWER, DOC, random.Random(1))
    assert case is not None, name
    assert case.answer != ANSWER
    verdict = verify(case.answer, DOC)
    assert not verdict.accepted
    assert verdict.codes & case.expected


def test_faults_that_would_still_be_true_are_skipped():
    doc = Document.build("T", [["", "a"], ["x", "5 and -5"]], [])
    answer = Answer(
        evidence=[Evidence(id="e1", value=5, source=TableSource(row=1, col=1))],
        plan="e1 * 2",
        answer_unit="number",
    )
    assert FAULTS["sign_flip"](answer, doc, random.Random(0)) is None


def test_runner_reports_rates():
    class Ex:
        target, document = ANSWER, DOC

    results = run_output_faults([Ex()], seed=0)
    assert set(results) == set(FAULTS)
    assert all(r["detection_rate"] == 1.0 for r in results.values())
