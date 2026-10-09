from pathlib import Path

import pytest

from ledgermind.data.convert import ConversionError, convert, convert_all
from ledgermind.data.finqa import FinQAExample, load_file
from ledgermind.document import Document
from ledgermind.dsl import execute
from ledgermind.schema import TableSource, TextSource, parse_model_output, to_json

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"

DOC = Document.build(
    "TEST/2020/page_1.pdf",
    table=[
        ["", "2020", "2019", "2018"],
        ["net revenue", "$ 5829", "$ 5735", "$ 5600"],
        ["other expense", "$ -23158 ( 23158 )", "$ -9457 ( 9457 )", "$ -100 ( 100 )"],
        ["margin", "12% ( 12 % )", "10% ( 10 % )", "8% ( 8 % )"],
    ],
    sentences=["total headcount was 1200 at year end .", "the tax rate was 21% ( 21 % ) ."],
)


def make(program: str, gold: float | bool = 0.0, gold_inds: dict | None = None) -> FinQAExample:
    return FinQAExample("TEST-1", "what is it?", DOC, program, gold, gold_inds or {})


def run(converted) -> float | bool:
    t = converted.target
    return execute(t.plan, {e.id: e.value for e in t.evidence})


@pytest.fixture(scope="module")
def fixture_examples():
    return load_file(FIXTURE)


def test_fixture_examples_convert_and_reproduce_gold(fixture_examples):
    converted, report = convert_all(fixture_examples)
    assert report.coverage == 1.0
    for c in converted:
        value = run(c)
        if isinstance(c.gold_answer, bool):
            assert value is c.gold_answer
        else:
            assert value == pytest.approx(c.gold_answer, rel=1e-4, abs=1e-5)


def test_targets_round_trip_through_the_schema(fixture_examples):
    for c in convert_all(fixture_examples)[0]:
        assert parse_model_output(to_json(c.target)).output == c.target


def test_simple_ratio_and_evidence_reuse():
    c = convert(make("subtract(5829, 5735), divide(#0, 5735)"))
    assert c.target.plan == "(e1 - e2) / e2"
    assert [e.value for e in c.target.evidence] == [5829, 5735]
    assert c.target.evidence[0].source == TableSource(row=1, col=1)
    assert c.target.evidence[0].label == "net revenue (2020)"


def test_precedence_keeps_right_operand_grouped():
    c = convert(make("add(5735, 5600), subtract(5829, #0)"))
    assert c.target.plan == "e3 - (e1 + e2)"
    assert run(c) == 5829 - (5735 + 5600)


def test_percent_argument_becomes_explicit_division():
    c = convert(make("multiply(5829, 21%)"))
    assert c.target.plan == "e1 * (e2 / 100)"
    assert c.target.evidence[1].source == TextSource(sent=1)
    assert c.target.evidence[1].value == 21
    assert "percent_argument" in c.flags


def test_magnitude_of_negative_cell_becomes_explicit_negation():
    c = convert(make("add(23158, 9457)"))
    assert c.target.plan == "(-e1) + (-e2)"
    assert [e.value for e in c.target.evidence] == [-23158, -9457]
    assert run(c) == 23158 + 9457
    assert "sign_flip" in c.flags


def test_table_operations_expand_the_row():
    c = convert(make("table_average(net revenue, none)"))
    assert c.target.plan == "mean(e1, e2, e3)"
    assert run(c) == pytest.approx((5829 + 5735 + 5600) / 3)


def test_table_operations_over_percent_cells_use_fractions():
    c = convert(make("table_max(margin, none)"))
    assert c.target.plan == "max((e1 / 100), (e2 / 100), (e3 / 100))"
    assert run(c) == pytest.approx(0.12)


def test_constants_and_answer_units():
    c = convert(make("divide(5829, 5735), multiply(#0, const_100)"))
    assert c.target.plan == "e1 / e2 * 100"
    assert c.target.answer_unit == "percent"
    assert convert(make("greater(5829, 5735)")).target.answer_unit == "boolean"
    assert convert(make("divide(5829, const_1000)")).target.plan == "e1 / 1000"


def test_gold_supporting_facts_break_ties():
    doc_tie = Document.build("T", [["", "a"], ["x", "100"], ["y", "100"]], [])
    ex = FinQAExample("T-1", "q", doc_tie, "add(100, const_1)", 101.0, {"table_2": "y a 100"})
    c = convert(ex)
    assert c.target.evidence[0].source == TableSource(row=2, col=1)
    assert "ambiguous_location" in c.flags


def test_out_of_range_gold_indices_are_ignored():
    c = convert(make("add(1200, const_1)", gold_inds={"text_-1": "", "table_99": ""}))
    assert c.target.evidence[0].source == TextSource(sent=0)


@pytest.mark.parametrize(
    "program, reason",
    [
        ("add(4242, 5829)", "unresolved_number"),
        ("table_sum(no such row, none)", "unresolved_row"),
        ("divide(const_100, const_3)", "no_evidence"),
        ("add(5829, #3)", "bad_reference"),
        ("frobnicate(5829, 5735)", "unknown_op"),
        ("add 5829 5735", "bad_program"),
    ],
)
def test_failures_report_a_reason(program, reason):
    with pytest.raises(ConversionError) as info:
        convert(make(program))
    assert info.value.reason == reason
