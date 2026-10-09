"""GRPO reward built on the verifier (reinforcement learning with verifiable rewards).

There is no learned reward model. Every term comes from a deterministic check, so the
reward cannot be flattered by fluent but wrong output, and each term is logged
separately during training so that any reward hacking shows up in the term that moves.

==================  =======  ========================================================
term                weight   when
==================  =======  ========================================================
correct             +1.0     executed answer matches gold, with no fabricated figure
                             and no bypass
provenance          +0.3     every evidence item the plan uses passes provenance
schema              +0.1     output parses as an Answer or an Abstain
fabrication         -0.5     any cited figure is fabricated, misattributed or dangling
bypass              -1.0     plan does not compute from evidence, or the output
                             smuggles in a final answer field
unused_evidence     -0.05    per cited item the plan never uses (capped at -0.25)
invariant           -0.2     unit, scale or period mismatch
==================  =======  ========================================================

The resulting order of outcomes is: correct and grounded (1.4) > grounded but wrong
(0.4) > abstain (0.1) > unparseable (0) > fabricated (at most -0.4) > bypass (-0.9).
The provenance bonus only counts evidence the plan uses, and unused evidence costs
reward, so padding an answer with easy, real figures does not pay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ledgermind.document import Document
from ledgermind.eval.answer import MatchMode, answers_match
from ledgermind.schema import Answer, first_json_object, parse_model_output
from ledgermind.verifier import Verdict, verify
from ledgermind.verifier.integrity import IntegrityIssue

SMUGGLED_KEYS = frozenset({"answer", "final_answer", "result", "value", "final_value"})
_INVARIANT_CODES = frozenset({"unit_mismatch", "scale_mismatch", "period_mismatch"})
_FABRICATION_CODES = frozenset({"fabricated", "misattributed", "bad_pointer"})


@dataclass(frozen=True)
class RewardWeights:
    correct: float = 1.0
    provenance: float = 0.3
    schema: float = 0.1
    fabrication: float = -0.5
    bypass: float = -1.0
    unused_evidence: float = -0.05
    unused_evidence_cap: float = -0.25
    invariant: float = -0.2


DEFAULT_WEIGHTS = RewardWeights()


@dataclass
class RewardBreakdown:
    total: float
    terms: dict[str, float] = field(default_factory=dict)
    verdict: Verdict | None = None


def _smuggled(text: str) -> bool:
    obj = first_json_object(text)
    return obj is not None and bool(SMUGGLED_KEYS & obj.keys())


def compute_reward(
    completion: str,
    doc: Document,
    gold: float | bool,
    *,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    match_mode: MatchMode = "scale",
    integrity_issues: list[IntegrityIssue] | None = None,
) -> RewardBreakdown:
    terms = dict.fromkeys(
        (
            "correct",
            "provenance",
            "schema",
            "fabrication",
            "bypass",
            "unused_evidence",
            "invariant",
        ),
        0.0,
    )
    parsed = parse_model_output(completion)
    if not parsed.ok:
        if _smuggled(completion):
            terms["bypass"] = weights.bypass
        return RewardBreakdown(sum(terms.values()), terms)

    terms["schema"] = weights.schema
    verdict = verify(parsed, doc, integrity_issues=integrity_issues)
    if not isinstance(parsed.output, Answer):
        return RewardBreakdown(sum(terms.values()), terms, verdict)

    answer = parsed.output
    codes = verdict.codes
    used = {e.id for e in answer.evidence} - set(verdict.unused_evidence)
    checks = {c.evidence_id: c for c in verdict.evidence_checks}
    fabricated = bool(codes & _FABRICATION_CODES)
    bypassed = "bypass" in codes

    if used and all(checks[i].ok for i in used):
        terms["provenance"] = weights.provenance
    if fabricated:
        terms["fabrication"] = weights.fabrication
    if bypassed:
        terms["bypass"] = weights.bypass
    if codes & _INVARIANT_CODES:
        terms["invariant"] = weights.invariant
    terms["unused_evidence"] = max(
        weights.unused_evidence * len(verdict.unused_evidence), weights.unused_evidence_cap
    )
    if not fabricated and not bypassed and answers_match(verdict.value, gold, match_mode):
        terms["correct"] = weights.correct
    return RewardBreakdown(sum(terms.values()), terms, verdict)
