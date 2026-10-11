"""Async client for OpenAI compatible endpoints, with per request timing.

Works against a vLLM server (``/v1/completions`` for our plain text prompts) and against
API baselines (``/v1/chat/completions``). Optional constrained decoding sends the output
JSON schema as ``response_format``, which vLLM enforces token by token.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from ledgermind.schema import output_json_schema

Endpoint = Literal["completions", "chat"]


@dataclass(frozen=True)
class Generation:
    id: str
    completion: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None = None

    def as_record(self) -> dict:
        return {
            "id": self.id,
            "completion": self.completion,
            "latency_s": round(self.latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


@dataclass
class Budget:
    """Hard spending cap for paid APIs, enforced before each request is sent.

    Cost comes from the token usage each response reports times the list prices. A request
    starts only if the spend so far plus every in flight request, each priced at the most
    expensive request seen, stays within the cap, so the run stops short of the cap rather
    than past it.
    """

    cap_usd: float
    usd_per_million_input: float
    usd_per_million_output: float
    spent_usd: float = 0.0
    in_flight: int = 0
    max_request_usd: float = 0.0
    sent: int = 0
    skipped: int = 0

    def cost(self, g: Generation) -> float:
        tokens_in, tokens_out = g.prompt_tokens or 0, g.completion_tokens or 0
        return (
            tokens_in * self.usd_per_million_input + tokens_out * self.usd_per_million_output
        ) / 1e6

    def try_start(self) -> bool:
        projected = self.spent_usd + (self.in_flight + 1) * self.max_request_usd
        if self.spent_usd >= self.cap_usd or projected > self.cap_usd:
            self.skipped += 1
            return False
        self.in_flight += 1
        self.sent += 1
        return True

    def finish(self, g: Generation) -> None:
        self.in_flight -= 1
        c = self.cost(g)
        self.spent_usd += c
        self.max_request_usd = max(self.max_request_usd, c)


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    model: str
    endpoint: Endpoint = "completions"
    api_key: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int | None = None  # per request sampling seed, for reproducible sampled evals
    # OpenAI reasoning models take max_completion_tokens (reasoning included) and reject a
    # custom temperature or seed; effort None keeps the model default.
    reasoning: bool = False
    reasoning_effort: str | None = None
    constrained: bool = False
    timeout_s: float = 300.0
    retries: int = 2


def request_body(cfg: ClientConfig, prompt: str) -> dict:
    if cfg.reasoning:
        body: dict = {"model": cfg.model, "max_completion_tokens": cfg.max_tokens}
        if cfg.reasoning_effort:
            body["reasoning_effort"] = cfg.reasoning_effort
    else:
        body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature}
        if cfg.seed is not None:
            body["seed"] = cfg.seed
    if cfg.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if cfg.constrained:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ledgermind_output", "schema": output_json_schema()},
        }
    return body


def _text(cfg: ClientConfig, payload: dict) -> str:
    choice = payload["choices"][0]
    # A reasoning model that spends its whole budget thinking returns content null
    text = choice["message"]["content"] if cfg.endpoint == "chat" else choice["text"]
    return text or ""


async def generate_one(
    client: httpx.AsyncClient, cfg: ClientConfig, item_id: str, prompt: str
) -> Generation:
    path = "/chat/completions" if cfg.endpoint == "chat" else "/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    error = None
    for attempt in range(cfg.retries + 1):
        start = time.perf_counter()
        try:
            response = await client.post(
                cfg.base_url.rstrip("/") + path, json=request_body(cfg, prompt), headers=headers
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            return Generation(
                item_id,
                _text(cfg, payload),
                time.perf_counter() - start,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.retries:
                await asyncio.sleep(2**attempt)
    return Generation(item_id, "", 0.0, None, None, error)


async def generate_all(
    cfg: ClientConfig,
    items: list[tuple[str, str]],
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None = None,
    budget: Budget | None = None,
) -> tuple[list[Generation], float]:
    """Run all prompts with at most ``concurrency`` in flight. Returns results and wall time.

    With a ``budget``, prompts that would exceed the cap are not sent and come back with an
    error, so the output still has one record per prompt.
    """
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        timeout=cfg.timeout_s, limits=limits, transport=transport
    ) as client:

        async def bounded(item_id: str, prompt: str) -> Generation:
            async with semaphore:
                if budget is None:
                    return await generate_one(client, cfg, item_id, prompt)
                if not budget.try_start():
                    return Generation(item_id, "", 0.0, None, None, "budget exhausted: not sent")
                g = await generate_one(client, cfg, item_id, prompt)
                budget.finish(g)
                return g

        start = time.perf_counter()
        results = await asyncio.gather(*(bounded(i, p) for i, p in items))
        return list(results), time.perf_counter() - start
