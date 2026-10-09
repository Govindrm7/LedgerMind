"""The structured output contract between the model and the verifier.

The model emits exactly one JSON object, either an :class:`Answer` (evidence with source
pointers plus a computation plan) or an :class:`Abstain`. There is deliberately no field
for a final number: the executor computes it. ``extra="forbid"`` means an output that
tries to smuggle in ``"answer": 42`` fails validation instead of being silently accepted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

REASONING_MAX_CHARS = 600
MAX_EVIDENCE = 32
PLAN_MAX_CHARS = 500

AnswerUnit = Literal["number", "percent", "currency", "ratio", "boolean"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Pointer(BaseModel):
    """Immutable and hashable, so pointers can key sets and dicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TableSource(_Pointer):
    """A table cell, addressed by zero based row and column (row 0 is the header)."""

    type: Literal["table"] = "table"
    row: int = Field(ge=0)
    col: int = Field(ge=0)


class TextSource(_Pointer):
    """A sentence of the document text, addressed by zero based index."""

    type: Literal["text"] = "text"
    sent: int = Field(ge=0)


Source = Annotated[TableSource | TextSource, Field(discriminator="type")]


class Evidence(_Strict):
    id: str = Field(pattern=r"^e[1-9][0-9]*$")
    value: float
    label: str = Field(default="", max_length=200)
    source: Source

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("evidence value must be finite")
        return v


class Answer(_Strict):
    reasoning: str = Field(default="", max_length=REASONING_MAX_CHARS)
    evidence: list[Evidence] = Field(min_length=1, max_length=MAX_EVIDENCE)
    plan: str = Field(min_length=1, max_length=PLAN_MAX_CHARS)
    answer_unit: AnswerUnit

    @model_validator(mode="after")
    def _unique_ids(self) -> Answer:
        ids = [e.id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique")
        return self

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}


class Abstain(_Strict):
    abstain: Literal[True]
    reason: str = Field(min_length=1, max_length=REASONING_MAX_CHARS)


ModelOutput = Answer | Abstain


def to_json(output: ModelOutput) -> str:
    """Canonical single line serialization, used for SFT targets."""
    return json.dumps(output.model_dump(mode="json"), ensure_ascii=False)


@dataclass(frozen=True)
class ParseResult:
    output: ModelOutput | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.output is not None


def first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def parse_model_output(text: str) -> ParseResult:
    """Parse raw model text into a validated output.

    Leading prose or code fences are tolerated, so that the unconstrained parse rate
    measures schema competence rather than formatting noise. Only the first JSON
    object is considered.
    """
    obj = first_json_object(text)
    if obj is None:
        return ParseResult(None, "no JSON object found")
    model: type[Answer] | type[Abstain] = Abstain if "abstain" in obj else Answer
    try:
        return ParseResult(model.model_validate(obj))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        return ParseResult(None, f"schema violation at {loc}: {first['msg']}")
