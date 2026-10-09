import random

import pytest

from ledgermind.document import Document
from ledgermind.saboteur.document_faults import DOCUMENT_FAULTS
from ledgermind.saboteur.run import run_document_faults
from ledgermind.schema import Answer, Evidence, TableSource
from ledgermind.verifier import verify
from ledgermind.verifier.integrity import footed_cells

DOC = Document.build(
    "T",
    table=[
        ["( in millions )", "2020", "2019"],
        ["product revenue", "$ 1200", "$ 1100"],
        ["service revenue", "300", "250"],
        ["total revenue", "$ 1500", "$ 1350"],
    ],
    sentences=[],
)
ANSWER = Answer(
    evidence=[
        Evidence(id="e1", value=300, source=TableSource(row=2, col=1)),
        Evidence(id="e2", value=1500, source=TableSource(row=3, col=1)),
    ],
    plan="e1 / e2",
    answer_unit="ratio",
)


@pytest.mark.parametrize("name", sorted(DOCUMENT_FAULTS))
def test_stale_answers_are_rejected(name):
    fault = DOCUMENT_FAULTS[name](ANSWER, DOC, random.Random(3))
    assert fault is not None
    assert fault.document != DOC
    assert not verify(ANSWER, fault.document).accepted


@pytest.mark.parametrize("name", ["digit_change", "sign_flip", "swap_columns"])
def test_faithful_reader_passes_provenance(name):
    fault = DOCUMENT_FAULTS[name](ANSWER, DOC, random.Random(3))
    verdict = verify(fault.faithful, fault.document, integrity=False)
    assert not {"fabricated", "misattributed"} & verdict.codes


def test_footing_flags_a_faithful_reader_of_a_tampered_line_item():
    tampered = DOC.with_cell(2, 1, "390")
    faithful = ANSWER.model_copy(
        update={
            "evidence": [ANSWER.evidence[0].model_copy(update={"value": 390}), ANSWER.evidence[1]]
        }
    )
    assert "source_footing_broken" in verify(faithful, tampered).codes


def test_footed_cells_cover_the_blocks_that_sum():
    cells = footed_cells(DOC)
    assert TableSource(row=1, col=1) in cells and TableSource(row=3, col=2) in cells


def test_runner_reports_all_faults():
    class Ex:
        target, document = ANSWER, DOC

    results = run_document_faults([Ex()], seed=0)
    assert set(results) == set(DOCUMENT_FAULTS)
    assert all(r["stale_rejection_rate"] == 1.0 for r in results.values())
