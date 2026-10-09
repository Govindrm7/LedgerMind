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


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    model: str
    endpoint: Endpoint = "completions"
    api_key: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int | None = None  # per request sampling seed, for reproducible sampled evals
    constrained: bool = False
    timeout_s: float = 300.0
    retries: int = 2


def request_body(cfg: ClientConfig, prompt: str) -> dict:
    body: dict = {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature}
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
    return choice["message"]["content"] if cfg.endpoint == "chat" else choice["text"]


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
) -> tuple[list[Generation], float]:
    """Run all prompts with at most ``concurrency`` in flight. Returns results and wall time."""
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        timeout=cfg.timeout_s, limits=limits, transport=transport
    ) as client:

        async def bounded(item_id: str, prompt: str) -> Generation:
            async with semaphore:
                return await generate_one(client, cfg, item_id, prompt)

        start = time.perf_counter()
        results = await asyncio.gather(*(bounded(i, p) for i, p in items))
        return list(results), time.perf_counter() - start
