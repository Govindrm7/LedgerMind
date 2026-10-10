import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from ledgermind.eval.cost import api_cost_per_1k, self_hosted_cost_per_1k
from ledgermind.schema import output_json_schema
from ledgermind.serving.bench import sweep
from ledgermind.serving.client import Budget, ClientConfig, Generation, generate_all, request_body

CFG = ClientConfig(base_url="http://server/v1", model="ledgermind", retries=1)


def fake_server(fail_first: int = 0):
    calls = {"n": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        calls["bodies"].append((request.url.path, body))
        if calls["n"] <= fail_first:
            return httpx.Response(503)
        text = "{}" if "prompt" in body else None
        choice = {"text": text} if text is not None else {"message": {"content": "Answer: 1"}}
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        return httpx.Response(200, json={"choices": [choice], "usage": usage})

    return httpx.MockTransport(handler), calls


def test_completion_requests_and_usage():
    transport, calls = fake_server()
    items = [(f"q{i}", f"prompt {i}") for i in range(5)]
    results, wall = asyncio.run(generate_all(CFG, items, concurrency=2, transport=transport))
    assert [g.id for g in results] == [f"q{i}" for i in range(5)]
    assert all(g.completion == "{}" and g.completion_tokens == 20 for g in results)
    path, body = calls["bodies"][0]
    assert path == "/v1/completions" and body["prompt"].startswith("prompt")
    assert body["temperature"] == 0.0 and wall >= 0


def test_chat_endpoint():
    transport, calls = fake_server()
    cfg = ClientConfig(base_url="http://server/v1", model="m", endpoint="chat")
    (g,), _ = asyncio.run(generate_all(cfg, [("q", "hi")], 1, transport))
    assert g.completion == "Answer: 1"
    assert calls["bodies"][0][0] == "/v1/chat/completions"


def test_retries_then_succeeds_and_reports_errors():
    transport, _ = fake_server(fail_first=1)
    (g,), _ = asyncio.run(generate_all(CFG, [("q", "p")], 1, transport))
    assert g.error is None
    transport, _ = fake_server(fail_first=10)
    (g,), _ = asyncio.run(generate_all(CFG, [("q", "p")], 1, transport))
    assert g.error and "503" in g.error


def test_constrained_decoding_sends_the_output_schema():
    cfg = ClientConfig(base_url="u", model="m", constrained=True)
    fmt = request_body(cfg, "p")["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == output_json_schema()
    assert "anyOf" in json.dumps(output_json_schema())


def test_sampling_seed_is_sent_only_when_set():
    assert "seed" not in request_body(CFG, "p")
    body = request_body(ClientConfig(base_url="u", model="m", temperature=0.9, seed=3), "p")
    assert body["temperature"] == 0.9 and body["seed"] == 3


def test_reasoning_models_get_completion_token_budget_and_no_temperature():
    cfg = ClientConfig(base_url="u", model="m", endpoint="chat", reasoning=True, max_tokens=8000)
    body = request_body(cfg, "p")
    assert body["max_completion_tokens"] == 8000
    assert not {"max_tokens", "temperature", "seed", "reasoning_effort"} & body.keys()
    effort = ClientConfig(base_url="u", model="m", reasoning=True, reasoning_effort="low")
    assert request_body(effort, "p")["reasoning_effort"] == "low"


def test_budget_stops_before_the_cap():
    # Each mock response reports 100 input and 20 output tokens: $0.003 at these prices.
    transport, calls = fake_server()
    budget = Budget(cap_usd=0.01, usd_per_million_input=10.0, usd_per_million_output=100.0)
    items = [(f"q{i}", "p") for i in range(10)]
    results, _ = asyncio.run(generate_all(CFG, items, 1, transport, budget=budget))
    assert calls["n"] == budget.sent == 3 and budget.skipped == 7
    assert budget.spent_usd == pytest.approx(0.009) and budget.spent_usd <= 0.01
    assert len(results) == 10
    assert sum(g.error == "budget exhausted: not sent" for g in results) == 7


def test_budget_with_concurrency_never_passes_the_cap_once_a_cost_is_known():
    transport, _ = fake_server()
    budget = Budget(cap_usd=0.02, usd_per_million_input=10.0, usd_per_million_output=100.0)
    items = [(f"q{i}", "p") for i in range(50)]
    asyncio.run(generate_all(CFG, items, 4, transport, budget=budget))
    assert budget.spent_usd <= 0.02 + 1e-12 and budget.skipped > 0


def test_budget_starting_at_the_cap_sends_nothing():
    transport, calls = fake_server()
    budget = Budget(0.01, 10.0, 100.0, spent_usd=0.01)
    asyncio.run(generate_all(CFG, [("q", "p")], 1, transport, budget=budget))
    assert calls["n"] == 0 and budget.skipped == 1


def test_predict_ledger_accumulates_across_runs(tmp_path, monkeypatch):
    from ledgermind.serving import predict

    async def fake_generate_all(cfg, items, concurrency, transport=None, budget=None):
        out = []
        for item_id, _ in items:
            assert budget.try_start()
            g = Generation(item_id, "{}", 0.1, 1000, 100)
            budget.finish(g)
            out.append(g)
        return out, 0.1

    examples = [SimpleNamespace(id=f"q{i}", question="q", document=None) for i in range(2)]
    monkeypatch.setattr(predict, "generate_all", fake_generate_all)
    monkeypatch.setattr(predict, "read_jsonl", lambda path: examples)
    monkeypatch.setattr(predict, "build_prompt", lambda question, document, version="v1": "p")
    ledger = tmp_path / "spend.json"
    args = [
        "--examples",
        "unused.jsonl",
        "--base-url",
        "u",
        "--model",
        "m",
        "--max-cost-usd",
        "5",
        "--price-input",
        "1",
        "--price-output",
        "10",
        "--ledger",
        str(ledger),
        "--limit",
        "2",
    ]
    for run in ("a", "b"):
        predict.main([*args, "--out", str(tmp_path / f"{run}.jsonl")])
    saved = json.loads(ledger.read_text())
    assert len(saved["runs"]) == 2
    assert saved["spent_usd"] == pytest.approx(4 * (1000 * 1 + 100 * 10) / 1e6)


def test_bench_sweep_reports_each_level():
    transport, _ = fake_server()
    items = [(f"q{i}", "p") for i in range(8)]
    rows = asyncio.run(sweep(CFG, items, [1, 4], transport))
    assert [r["concurrency"] for r in rows] == [1, 4]
    assert all(r["requests"] == 8 and r["errors"] == 0 for r in rows)
    assert all(r["output_tokens_per_s"] > 0 for r in rows)


def test_costs():
    assert self_hosted_cost_per_1k(2.0, 10.0) == pytest.approx(2.0 / 36)
    assert api_cost_per_1k(2000, 200, 3.0, 15.0) == pytest.approx(9.0)
    with pytest.raises(ValueError):
        self_hosted_cost_per_1k(2.0, 0)
