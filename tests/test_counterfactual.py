import random

from ledgermind.data.convert import ConvertedExample
from ledgermind.document import Document
from ledgermind.dsl import execute
from ledgermind.saboteur.counterfactual import build_counterfactual, rewrite_document
from ledgermind.schema import Answer, Evidence, TableSource, TextSource
from ledgermind.verifier import verify

DOC = Document.build(
    "T",
    table=[
        ["", "2020", "2019"],
        ["net revenue", "$ 5829", "$ 5735"],
        ["net loss", "$ -9457 ( 9457 )", "$ -9000 ( 9000 )"],
    ],
    sentences=["net revenue rose to $ 5829 million in 2020 ."],
)
TARGET = Answer(
    evidence=[
        Evidence(id="e1", value=5829, source=TableSource(row=1, col=1)),
        Evidence(id="e2", value=5735, source=TableSource(row=1, col=2)),
        Evidence(id="e3", value=-9457, source=TableSource(row=2, col=1)),
    ],
    plan="(e1 - e2) / (-e3)",
    answer_unit="ratio",
)
GOLD = execute(TARGET.plan, {e.id: e.value for e in TARGET.evidence})
EXAMPLE = ConvertedExample("T-1", "q", DOC, TARGET, GOLD)


def test_counterfactual_changes_answer_and_stays_verifiable():
    cf = build_counterfactual(EXAMPLE, random.Random(0))
    assert cf is not None
    assert cf.original_gold == GOLD
    assert cf.example.gold_answer != GOLD
    assert verify(cf.example.target, cf.example.document, integrity=False).accepted
    assert not verify(TARGET, cf.example.document, integrity=False).accepted


def test_every_copy_of_an_edited_figure_is_rewritten():
    cf = build_counterfactual(EXAMPLE, random.Random(0))
    new_revenue = cf.example.target.evidence[0].value
    assert cf.example.document.find_value(5829) == []
    assert TextSource(sent=0) in cf.example.document.find_value(new_revenue)


def test_signs_and_echo_groups_are_preserved():
    cf = build_counterfactual(EXAMPLE, random.Random(0))
    loss = cf.example.document.table[2][1]
    new = cf.example.target.evidence[2].value
    assert new < 0
    digits = f"{abs(new):.0f}"
    assert loss == f"$ -{digits} ( {digits} )"


def test_rewrite_document_leaves_other_figures_alone():
    doc = rewrite_document(DOC, {5829.0: "6000"})
    assert doc.table[1][2] == "$ 5735"
    assert doc.table[1][1] == "$ 6000"
