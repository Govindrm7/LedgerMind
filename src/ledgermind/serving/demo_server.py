"""Live demo: ask the served model a FinQA question and show the verifier's audit trail.

A small standard library HTTP server. It builds the same prompt the model was trained
on, sends it to a vLLM server, runs the LedgerMind verifier on the reply, and returns
the audit trail that the demo page renders (``serving/demo``). Meant for a short lived
GPU job reached through an SSH tunnel, so it binds to localhost by default.

Usage::

    python -m ledgermind.serving.demo_server --vllm http://localhost:8000/v1 --port 9000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.data.prompts import build_prompt
from ledgermind.eval.answer import answers_match
from ledgermind.verifier.verifier import verify

ROOT = Path(__file__).resolve().parents[3]
STATIC = {
    "/": (ROOT / "serving" / "demo" / "index.html", "text/html"),
    "/static/style.css": (ROOT / "docs" / "dashboard" / "style.css", "text/css"),
    "/static/audit.js": (ROOT / "docs" / "dashboard" / "audit.js", "text/javascript"),
    "/static/demo.js": (ROOT / "serving" / "demo" / "demo.js", "text/javascript"),
}
MAX_QUESTION_CHARS = 500


def load_examples(data_dir: Path, ids_file: Path | None, limit: int) -> list[dict]:
    """Demo questions: the dashboard explorer's selection when available, else the first N."""
    test = {e.id: e for e in read_jsonl(data_dir / "test.jsonl")}
    cf_path = data_dir / "counterfactual_test.jsonl"
    cf = {e.id: e for e in read_jsonl(cf_path)} if cf_path.exists() else {}
    wanted: list[tuple[str, str]] = []
    if ids_file and ids_file.exists():
        explorer = json.loads(ids_file.read_text())["explorer"]
        wanted = [
            (q["id"], "counterfactual" if q["group"] == "counterfactual" else "test")
            for q in explorer
        ]
    if not wanted:
        wanted = [(i, "test") for i in list(test)[:limit]]
    out = []
    for i, source in wanted[:limit]:
        ex = (cf if source == "counterfactual" else test).get(i)
        if ex is None:
            continue
        out.append(
            {
                "key": f"{source}:{i}",
                "id": i,
                "source": source,
                "question": ex.question,
                "gold": ex.gold_answer,
                "original_gold": test[i].gold_answer if source == "counterfactual" else None,
                "document": {
                    "table": [list(r) for r in ex.document.table],
                    "sentences": list(ex.document.sentences),
                },
                "_example": ex,
            }
        )
    return out


def gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10
        )
        return out.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown GPU"


class DemoState:
    def __init__(self, vllm_url: str, model: str, examples: list[dict], label: str):
        self.vllm_url = vllm_url.rstrip("/")
        self.model = model
        self.examples = {e["key"]: e for e in examples}
        self.order = [e["key"] for e in examples]
        self.info = {"model_label": label, "gpu": gpu_name(), "examples": len(examples)}
        # The cluster sets an HTTP proxy; requests to the local vLLM server must bypass it.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ask(self, key: str, question: str | None) -> dict:
        ex = self.examples[key]
        original = ex["_example"]
        q = (question or original.question).strip()[:MAX_QUESTION_CHARS]
        body = {
            "model": self.model,
            "prompt": build_prompt(q, original.document),
            "max_tokens": 512,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{self.vllm_url}/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with self.opener.open(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        latency = time.perf_counter() - start
        completion = payload["choices"][0]["text"]
        verdict = verify(completion, original.document)
        same_question = q == original.question
        return {
            "question": q,
            "audit": verdict.audit(),
            "raw": completion,
            "latency_s": round(latency, 3),
            "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            "gold": original.gold_answer if same_question else None,
            "correct": bool(
                verdict.accepted and answers_match(verdict.value, original.gold_answer, "scale")
            )
            if same_question
            else None,
        }


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state: DemoState, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path in STATIC:
            path, ctype = STATIC[self.path]
            self._send(200, path.read_bytes(), ctype)
        elif self.path == "/api/info":
            self._json(200, self.state.info)
        elif self.path == "/api/examples":
            keep = ("key", "id", "source", "question", "gold", "original_gold", "document")
            self._json(
                200, [{k: self.state.examples[key][k] for k in keep} for key in self.state.order]
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._json(404, {"error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 10_000)
            req = json.loads(self.rfile.read(length) or b"{}")
            if req.get("key") not in self.state.examples:
                self._json(400, {"error": "unknown example"})
                return
            self._json(200, self.state.ask(req["key"], req.get("question")))
        except Exception as exc:  # the demo reports failures instead of dropping the request
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[demo] {self.address_string()} {fmt % args}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vllm", required=True, help="OpenAI compatible base URL of the model server"
    )
    parser.add_argument("--model", default="ledgermind", help="served model name")
    parser.add_argument("--label", default="LedgerMind GRPO, Qwen3-4B with LoRA")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data", type=Path, default=Path("data/processed/finqa"))
    parser.add_argument("--examples-from", type=Path, default=Path("docs/dashboard/data.json"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)

    examples = load_examples(args.data, args.examples_from, args.limit)
    state = DemoState(args.vllm, args.model, examples, args.label)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, state=state))
    print(
        f"demo on http://{args.host}:{args.port}, {len(examples)} examples, {state.info['gpu']}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
