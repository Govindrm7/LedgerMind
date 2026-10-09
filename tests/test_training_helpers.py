import json

from ledgermind.data.convert import ConvertedExample
from ledgermind.data.prompts import build_prompt
from ledgermind.document import Document
from ledgermind.schema import Answer, Evidence, TableSource, to_json
from ledgermind.training.pass_rate import informative_prompts, pass_rates, summarize_rates
from ledgermind.training.records import document_from_json, grpo_record, sft_record
from ledgermind.training.reward_fn import make_reward_fn

DOC = Document.build("T", [["", "2020", "2019"], ["revenue", "$ 5829", "$ 5735"]], [])
TARGET = Answer(
    evidence=[
        Evidence(id="e1", value=5829, source=TableSource(row=1, col=1)),
        Evidence(id="e2", value=5735, source=TableSource(row=1, col=2)),
    ],
    plan="(e1 - e2) / e2",
    answer_unit="ratio",
)
EX = ConvertedExample("T-1", "growth?", DOC, TARGET, (5829 - 5735) / 5735)


def test_sft_record_is_prompt_completion():
    rec = sft_record(EX)
    assert rec["prompt"] == build_prompt("growth?", DOC)
    assert json.loads(rec["completion"]) == json.loads(to_json(TARGET))


def test_grpo_record_round_trips_document_and_gold():
    rec = grpo_record(EX)
    assert document_from_json(rec["document_json"]) == DOC
    assert json.loads(rec["gold_json"]) == EX.gold_answer
    boolean = ConvertedExample("T-2", "q", DOC, TARGET, True)
    assert json.loads(grpo_record(boolean)["gold_json"]) is True


def test_reward_fn_matches_trl_interface_and_logs_terms():
    rec = grpo_record(EX)
    logged = {}
    fn = make_reward_fn()
    rewards = fn(
        prompts=[rec["prompt"]] * 3,
        completions=[
            to_json(TARGET),
            "garbage",
            [{"role": "assistant", "content": to_json(TARGET)}],
        ],
        document_json=[rec["document_json"]] * 3,
        gold_json=[rec["gold_json"]] * 3,
        completion_ids=[[1], [2], [3]],
        trainer_state=None,
        log_metric=lambda name, value: logged.__setitem__(name, value),
    )
    assert rewards == [1.4, 0.0, 1.4]
    assert logged["reward/correct"] == 2 / 3
    assert logged["verdict/invalid"] == 1 / 3


def test_pass_rate_filtering():
    rates = pass_rates({"a": [True] * 8, "b": [False] * 8, "c": [True, False] * 4, "d": []})
    assert rates == {"a": 1.0, "b": 0.0, "c": 0.5}
    assert informative_prompts(rates) == ["c"]
    summary = summarize_rates(rates)
    assert summary["informative"] == 1 and summary["always_solved"] == 1
