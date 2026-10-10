import json
import threading
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ledgermind.data.convert import ConvertedExample
from ledgermind.document import Document
from ledgermind.schema import Answer, Evidence, TableSource, to_json
from ledgermind.serving.demo_server import DemoState, Handler

DOC = Document.build(
    "T", [["", "2020", "2019"], ["revenue", "$ 5829", "$ 5735"]], ["revenue grew ."]
)
ANSWER = Answer(
    evidence=[
        Evidence(id="e1", value=5829, source=TableSource(row=1, col=1)),
        Evidence(id="e2", value=5735, source=TableSource(row=1, col=2)),
    ],
    plan="(e1 - e2) / e2",
    answer_unit="ratio",
)
GOLD = (5829 - 5735) / 5735


def serve(handler_cls, **kwargs):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(handler_cls, **kwargs) if kwargs else handler_cls
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class FakeVLLM(BaseHTTPRequestHandler):
    prompts: list = []
    reply = to_json(ANSWER)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.prompts.append(body)
        data = json.dumps(
            {"choices": [{"text": FakeVLLM.reply}], "usage": {"completion_tokens": 42}}
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def demo():
    vllm, vllm_url = serve(FakeVLLM)
    example = {
        "key": "test:T-0",
        "id": "T-0",
        "source": "test",
        "question": "growth?",
        "gold": GOLD,
        "original_gold": None,
        "document": {"table": [], "sentences": []},
        "_example": ConvertedExample("T-0", "growth?", DOC, ANSWER, GOLD),
    }
    state = DemoState(f"{vllm_url}/v1", "ledgermind", [example], "test model")
    app, app_url = serve(Handler, state=state)
    yield app_url
    app.shutdown()
    vllm.shutdown()


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_ask_returns_the_verified_audit_trail(demo):
    FakeVLLM.reply, FakeVLLM.prompts = to_json(ANSWER), []
    status, res = post(f"{demo}/api/ask", {"key": "test:T-0"})
    assert status == 200 and res["audit"]["status"] == "accepted"
    assert res["correct"] is True and res["gold"] == pytest.approx(GOLD)
    assert res["completion_tokens"] == 42
    sent = FakeVLLM.prompts[-1]
    assert sent["temperature"] == 0.0 and "### Question\ngrowth?" in sent["prompt"]


def test_a_fabricated_figure_is_rejected(demo):
    bad = ANSWER.model_copy(
        update={
            "evidence": [ANSWER.evidence[0].model_copy(update={"value": 5900}), ANSWER.evidence[1]]
        }
    )
    FakeVLLM.reply = to_json(bad)
    _, res = post(f"{demo}/api/ask", {"key": "test:T-0"})
    assert res["audit"]["status"] == "rejected" and res["correct"] is False
    FakeVLLM.reply = to_json(ANSWER)


def test_custom_question_has_no_gold(demo):
    _, res = post(f"{demo}/api/ask", {"key": "test:T-0", "question": "what was revenue in 2020?"})
    assert res["gold"] is None and res["correct"] is None
    assert res["question"] == "what was revenue in 2020?"


def test_unknown_example_is_refused(demo):
    status, res = post(f"{demo}/api/ask", {"key": "nope"})
    assert status == 400 and "unknown" in res["error"]
