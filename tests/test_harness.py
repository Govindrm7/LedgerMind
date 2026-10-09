import json

import pytest

from ledgermind.data.convert import ConvertedExample
from ledgermind.document import Document
from ledgermind.eval.harness import evaluate, oracle_predictions, parse_direct_answer
from ledgermind.schema import Answer, Evidence, TableSource, to_json

DOC = Document.build(
    "T", [["", "2020", "2019"], ["revenue", "$ 5829", "$ 5735"]], ["revenue grew ."]
)
TARGET = Answer(
    evidence=[
        Evidence(id="e1", value=5829, source=TableSource(row=1, col=1)),
        Evidence(id="e2", value=5735, source=TableSource(row=1, col=2)),
    ],
    plan="(e1 - e2) / e2",
    answer_unit="ratio",
)
GOLD = (5829 - 5735) / 5735


def example(i: int) -> ConvertedExample:
    return ConvertedExample(f"T-{i}", "growth?", DOC, TARGET, GOLD)


EXAMPLES = [example(i) for i in range(4)]


def fabricated() -> str:
    bad = TARGET.model_copy(
        update={
            "evidence": [TARGET.evidence[0], TARGET.evidence[1].model_copy(update={"value": 5700})]
        }
    )
    return to_json(bad)


def test_oracle_scores_perfectly():
    summary, rows = evaluate(EXAMPLES, oracle_predictions(EXAMPLES))
    assert summary["verified_accuracy"]["mean"] == 1.0
    assert summary["coverage"]["mean"] == 1.0
    assert summary["fabrication_rate"] == 0.0
    assert all(r.accepted for r in rows)


def test_mixed_outcomes():
    preds = {
        "T-0": {"completion": to_json(TARGET), "latency_s": 0.2, "completion_tokens": 100},
        "T-1": {"completion": fabricated(), "latency_s": 0.4, "completion_tokens": 120},
        "T-2": {"completion": '{"abstain": true, "reason": "unsure"}'},
        "T-3": {"completion": "not json"},
    }
    summary, rows = evaluate(EXAMPLES, preds)
    assert summary["verified_accuracy"]["mean"] == 0.25
    assert summary["coverage"]["mean"] == 0.25
    assert summary["parse_rate"] == 0.75
    assert summary["abstain_rate"] == 0.25
    assert summary["fabrication_rate"] == 0.5
    assert summary["rejection_reasons"]["fabricated"] == 1
    assert summary["latency_s"]["p50"] == pytest.approx(0.3)
    assert rows[1].value is not None and not rows[1].accepted


def test_ungated_accuracy_counts_values_the_verifier_rejected():
    misplaced = TARGET.model_copy(
        update={
            "evidence": [
                TARGET.evidence[0].model_copy(update={"source": TableSource(row=1, col=2)}),
                TARGET.evidence[1].model_copy(update={"source": TableSource(row=1, col=1)}),
            ]
        }
    )
    summary, _ = evaluate(EXAMPLES[:1], {"T-0": {"completion": to_json(misplaced)}})
    assert summary["verified_accuracy"]["mean"] == 0.0
    assert summary["ungated_accuracy"]["mean"] == 1.0


def test_missing_predictions_count_as_wrong():
    summary, _ = evaluate(EXAMPLES, {})
    assert summary["verified_accuracy"]["mean"] == 0.0 and summary["parse_rate"] == 0.0


@pytest.mark.parametrize(
    "text, value",
    [
        ("Growth was modest.\nAnswer: 1.64%", 1.64),
        ("so the result is 0.0164", 0.0164),
        ("Final answer: yes", True),
        ("answer = -23,158", -23158),
        ("no idea", None),
    ],
)
def test_parse_direct_answer(text, value):
    assert parse_direct_answer(text) == value


def test_direct_scoring_and_counterfactual_recall():
    preds = {
        "T-0": {"completion": f"Answer: {GOLD * 100:.4f}%"},
        "T-1": {"completion": "Answer: 5"},
    }
    originals = {"T-0": 0.5, "T-1": 5.0}
    summary, rows = evaluate(EXAMPLES[:2], preds, direct=True, originals=originals)
    assert rows[0].correct_scale and not rows[1].correct_scale
    assert rows[1].recalled_original
    assert summary["recalled_original_rate"]["mean"] == 0.5
    json.dumps(summary)
