import asyncio
import json

import httpx
import pytest

from ledgermind.eval.cost import api_cost_per_1k, self_hosted_cost_per_1k
from ledgermind.schema import output_json_schema
from ledgermind.serving.bench import sweep
from ledgermind.serving.client import ClientConfig, generate_all, request_body

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
