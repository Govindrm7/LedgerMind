import pytest

from ledgermind.document import Document
from ledgermind.schema import Answer, Evidence, TableSource, TextSource
from ledgermind.verifier.provenance import check_evidence, check_provenance

DOC = Document.build(
    "T",
    table=[
        ["", "2020", "2019"],
        ["revenue", "$ 5829", "$ 5735"],
        ["loss", "$ -9457 ( 9457 )", "12%"],
    ],
    sentences=["headcount was 1200 and churn was 3.5% ."],
)


def ev(value, source, eid="e1"):
    return Evidence(id=eid, value=value, source=source)


@pytest.mark.parametrize(
    "value, source, status",
    [
        (5829, TableSource(row=1, col=1), "ok"),
        (-9457, TableSource(row=2, col=1), "ok"),
        (12, TableSource(row=2, col=2), "ok"),
        (3.5, TextSource(sent=0), "ok"),
        (5735, TableSource(row=1, col=1), "misattributed"),
        (9457, TableSource(row=2, col=1), "fabricated"),
        (0.12, TableSource(row=2, col=2), "fabricated"),
        (5830, TableSource(row=1, col=1), "fabricated"),
        (5829, TableSource(row=5, col=1), "bad_pointer"),
        (1200, TextSource(sent=3), "bad_pointer"),
    ],
)
def test_statuses(value, source, status):
    assert check_evidence(ev(value, source), DOC).status == status


def test_misattributed_reports_where_value_is_printed():
    check = check_evidence(ev(5735, TableSource(row=1, col=1)), DOC)
    assert check.found_at == (TableSource(row=1, col=2),)
    assert check.cited_text == "$ 5829"


def test_check_provenance_covers_every_item():
    answer = Answer(
        evidence=[ev(5829, TableSource(row=1, col=1)), ev(1, TextSource(sent=0), "e2")],
        plan="e1 + e2",
        answer_unit="number",
    )
    statuses = [c.status for c in check_provenance(answer, DOC)]
    assert statuses == ["ok", "fabricated"]
