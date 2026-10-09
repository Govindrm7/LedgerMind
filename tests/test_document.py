import pytest

from ledgermind.document import Document
from ledgermind.schema import TableSource, TextSource


@pytest.fixture
def doc() -> Document:
    return Document.build(
        "ADI/2009/page_49.pdf",
        table=[
            ["", "october 31 2009", "november 1 2008"],
            ["fair value of contracts", "$ 6427", "$ -23158 ( 23158 )"],
            ["after unfavorable movement", "$ 20132", "$ -9457 ( 9457 )"],
        ],
        sentences=[
            "foreign currency exposure is described in note 2i .",
            "if libor changes by 100 basis points , expense would change by $ 3.8 million .",
        ],
    )


def test_resolve_cells_and_sentences(doc):
    assert doc.resolve(TableSource(row=1, col=2)) == "$ -23158 ( 23158 )"
    assert doc.resolve(TextSource(sent=1)).startswith("if libor")


def test_out_of_range_pointers_resolve_to_none(doc):
    assert doc.resolve(TableSource(row=9, col=0)) is None
    assert doc.resolve(TableSource(row=1, col=7)) is None
    assert doc.resolve(TextSource(sent=5)) is None


def test_numbers_at(doc):
    assert [m.value for m in doc.numbers_at(TableSource(row=1, col=2))] == [-23158]
    assert [m.value for m in doc.numbers_at(TextSource(sent=1))] == [100, 3.8]


def test_find_value(doc):
    assert doc.find_value(-9457) == [TableSource(row=2, col=2)]
    assert doc.find_value(9457) == []
    assert doc.find_value(3.8) == [TextSource(sent=1)]


def test_labels_and_headers(doc):
    assert doc.row_label(2) == "after unfavorable movement"
    assert doc.column_header(1) == "october 31 2009"


def test_corruptions_return_new_documents(doc):
    changed = doc.with_cell(1, 1, "$ 6428")
    assert changed.resolve(TableSource(row=1, col=1)) == "$ 6428"
    assert doc.resolve(TableSource(row=1, col=1)) == "$ 6427"
    assert doc.without_row(1).n_rows == 2
    assert doc.with_sentence(0, "changed").sentences[0] == "changed"
    assert doc.sentences[0] != "changed"


def test_render_exposes_coordinates(doc):
    text = doc.render()
    assert "r1 | fair value of contracts | $ 6427 | $ -23158 ( 23158 )" in text
    assert "[s1] if libor" in text
    assert text.splitlines()[1] == "r | c0 | c1 | c2"
