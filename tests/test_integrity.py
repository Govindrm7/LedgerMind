from ledgermind.document import Document
from ledgermind.schema import TableSource, TextSource
from ledgermind.verifier.integrity import (
    check_footings,
    check_integrity,
    check_text_table_agreement,
)

TABLE = [
    ["( in millions )", "2020", "2019", "average price"],
    ["product revenue", "$ 1200", "$ 1100", "$ 10.5"],
    ["service revenue", "300", "250", "12.0"],
    ["other", "2014", "50", "9.0"],
    ["total revenue", "$ 1500", "$ 1400", "$ 11.1"],
]
SENTENCES = ["product revenue in 2020 was $ 1200 million ."]


def doc(table=TABLE, sentences=SENTENCES) -> Document:
    return Document.build("T", table, sentences)


def test_clean_document_has_no_issues():
    assert check_integrity(doc()) == []


def test_altered_line_item_breaks_its_column():
    tampered = doc().with_cell(2, 1, "310")
    (issue,) = check_footings(tampered)
    assert issue.code == "footing_broken" and issue.severity == "reject"
    assert TableSource(row=2, col=1) in issue.locations
    assert TableSource(row=4, col=1) in issue.locations
    assert TableSource(row=2, col=2) not in issue.locations


def test_altered_total_is_detected():
    (issue,) = check_footings(doc().with_cell(4, 2, "$ 1410"))
    assert TableSource(row=4, col=2) in issue.locations


def test_sign_flip_is_detected():
    assert check_footings(doc().with_cell(3, 2, "-50 ( 50 )"))


def test_finqa_dash_cells_count_as_nil():
    # "2014" is how FinQA renders an em dash; column 1 foots only if it is read as zero.
    assert check_footings(doc()) == []


def test_non_additive_columns_are_ignored():
    assert check_footings(doc().with_cell(4, 3, "$ 99.9")) == []


def test_nested_subtotals_foot():
    table = [
        ["", "2020", "2019"],
        ["us", "100", "90"],
        ["egypt", "20", "10"],
        ["other africa", "30", "40"],
        ["total africa", "50", "50"],
        ["total", "150", "140"],
    ]
    assert check_footings(doc(table)) == []
    assert check_footings(doc(table).with_cell(1, 1, "110"))


def test_changes_within_rounding_tolerance_are_undetectable():
    # Printed figures are rounded, so a total may legitimately differ from the sum of its
    # items by up to half a unit per item. A last digit change of 1 hides inside that.
    assert check_footings(doc().with_cell(2, 1, "301")) == []


def test_text_table_disagreement_is_a_warning():
    tampered = doc().with_cell(1, 1, "$ 1300").with_cell(4, 1, "$ 1600")
    (issue,) = check_text_table_agreement(tampered)
    assert issue.severity == "warn"
    assert issue.locations == (TextSource(sent=0), TableSource(row=1, col=1))


def test_days_of_month_are_not_compared():
    table = [["", "2020"], ["accrued interest expense", "$ 21"]]
    sentences = ["accrued interest expense was recorded on december 31 , 2020 ."]
    assert check_text_table_agreement(doc(table, sentences)) == []
