import json

from ledgermind.schema import Abstain, Answer, TableSource, TextSource, parse_model_output, to_json

VALID = {
    "reasoning": "Change in net revenue divided by the 2015 value.",
    "evidence": [
        {
            "id": "e1",
            "value": 5829,
            "label": "Net revenue 2016",
            "source": {"type": "table", "row": 2, "col": 1},
        },
        {
            "id": "e2",
            "value": 5735,
            "label": "Net revenue 2015",
            "source": {"type": "text", "sent": 4},
        },
    ],
    "plan": "(e1 - e2) / e2",
    "answer_unit": "ratio",
}


def test_valid_answer_parses():
    result = parse_model_output(json.dumps(VALID))
    assert result.ok
    out = result.output
    assert isinstance(out, Answer)
    assert isinstance(out.evidence[0].source, TableSource)
    assert isinstance(out.evidence[1].source, TextSource)
    assert set(out.evidence_by_id()) == {"e1", "e2"}


def test_prose_and_fences_around_json_are_tolerated():
    text = "Here is the result:\n```json\n" + json.dumps(VALID) + "\n```"
    assert parse_model_output(text).ok


def test_abstain_parses():
    result = parse_model_output('{"abstain": true, "reason": "Line item missing."}')
    assert isinstance(result.output, Abstain)


def test_smuggled_final_answer_is_rejected():
    bad = dict(VALID, answer=1.64)
    result = parse_model_output(json.dumps(bad))
    assert not result.ok
    assert "answer" in result.error


def test_duplicate_ids_rejected():
    bad = dict(VALID, evidence=[VALID["evidence"][0], VALID["evidence"][0]])
    assert not parse_model_output(json.dumps(bad)).ok


def test_empty_evidence_rejected():
    assert not parse_model_output(json.dumps(dict(VALID, evidence=[]))).ok


def test_bad_unit_and_bad_id_rejected():
    assert not parse_model_output(json.dumps(dict(VALID, answer_unit="dollars"))).ok
    ev = dict(VALID["evidence"][0], id="x1")
    assert not parse_model_output(json.dumps(dict(VALID, evidence=[ev]))).ok


def test_non_finite_value_rejected():
    text = json.dumps(VALID).replace("5829", "NaN", 1)
    assert not parse_model_output(text).ok


def test_garbage_reports_error():
    result = parse_model_output("the answer is 42")
    assert not result.ok and result.error == "no JSON object found"


def test_round_trip_is_stable():
    out = parse_model_output(json.dumps(VALID)).output
    again = parse_model_output(to_json(out)).output
    assert again == out
