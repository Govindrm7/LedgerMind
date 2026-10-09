"""The verifier: one decision, with an audit trail, for every model output.

This single function is used in three places so that they can never disagree: as the
gate at inference time, as the reward signal during GRPO, and as the scorer in the eval
harness.

Gates run in a fixed order and every failure is recorded, not just the first, so the
audit trail and the per term reward logs show everything that went wrong:

1. schema       the output parses into an Answer or an Abstain
2. provenance   each cited figure is printed at its cited location
3. hygiene      the plan compiles, uses only defined evidence, depends on it
4. execution    the executor computes the value
5. invariants   units, scales and periods are combined consistently
6. integrity    the cited source cells are not part of a broken footing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ledgermind.document import Document
from ledgermind.dsl import PlanError
from ledgermind.schema import Abstain, Answer, ModelOutput, ParseResult, parse_model_output
from ledgermind.verifier.hygiene import check_hygiene
from ledgermind.verifier.integrity import IntegrityIssue, check_integrity
from ledgermind.verifier.invariants import check_invariants
from ledgermind.verifier.provenance import EvidenceCheck, check_provenance

Status = Literal["accepted", "rejected", "abstained", "invalid"]


@dataclass(frozen=True)
class Reason:
    code: str
    detail: str
    evidence_id: str | None = None

    def as_dict(self) -> dict:
        out = {"code": self.code, "detail": self.detail}
        if self.evidence_id:
            out["evidence_id"] = self.evidence_id
        return out


@dataclass
class Verdict:
    status: Status
    output: ModelOutput | None = None
    value: float | bool | None = None
    reasons: list[Reason] = field(default_factory=list)
    warnings: list[Reason] = field(default_factory=list)
    evidence_checks: list[EvidenceCheck] = field(default_factory=list)
    unused_evidence: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def codes(self) -> set[str]:
        return {r.code for r in self.reasons}

    def audit(self) -> dict:
        """JSON serializable trail of what was claimed, what was checked and the decision."""
        trail: dict = {
            "status": self.status,
            "value": self.value,
            "reasons": [r.as_dict() for r in self.reasons],
            "warnings": [w.as_dict() for w in self.warnings],
        }
        if isinstance(self.output, Answer):
            checks = {c.evidence_id: c for c in self.evidence_checks}
            trail["plan"] = self.output.plan
            trail["answer_unit"] = self.output.answer_unit
            trail["evidence"] = [
                {
                    "id": e.id,
                    "value": e.value,
                    "label": e.label,
                    "source": e.source.model_dump(),
                    "check": checks[e.id].status if e.id in checks else None,
                    "cited_text": checks[e.id].cited_text if e.id in checks else None,
                    "used": e.id not in self.unused_evidence,
                }
                for e in self.output.evidence
            ]
        elif isinstance(self.output, Abstain):
            trail["abstain_reason"] = self.output.reason
        return trail


def _integrity_reasons(
    answer: Answer, issues: list[IntegrityIssue]
) -> tuple[list[Reason], list[Reason]]:
    reasons: list[Reason] = []
    warnings: list[Reason] = []
    for issue in issues:
        touched = [e.id for e in answer.evidence if e.source in issue.locations]
        if not touched:
            continue
        target = reasons if issue.severity == "reject" else warnings
        for eid in touched:
            target.append(Reason(f"source_{issue.code}", issue.detail, eid))
    return reasons, warnings


def verify(
    output: str | ParseResult | ModelOutput,
    doc: Document,
    *,
    integrity: bool = True,
    integrity_issues: list[IntegrityIssue] | None = None,
) -> Verdict:
    """Decide whether an output may be answered. ``integrity_issues`` may be precomputed."""
    if isinstance(output, str):
        output = parse_model_output(output)
    if isinstance(output, ParseResult):
        if not output.ok:
            return Verdict("invalid", reasons=[Reason("schema_invalid", output.error or "")])
        output = output.output
    if isinstance(output, Abstain):
        return Verdict("abstained", output=output)

    answer: Answer = output  # type: ignore[assignment]
    verdict = Verdict("rejected", output=answer)

    verdict.evidence_checks = check_provenance(answer, doc)
    for check in verdict.evidence_checks:
        if not check.ok:
            detail = f"cited location reads {check.cited_text!r}"
            verdict.reasons.append(Reason(check.status, detail, check.evidence_id))

    hygiene = check_hygiene(answer)
    verdict.reasons.extend(Reason(code, detail) for code, detail in hygiene.errors)
    verdict.unused_evidence = hygiene.unused_evidence
    verdict.warnings.extend(
        Reason("unused_evidence", "cited but not used by the plan", eid)
        for eid in hygiene.unused_evidence
    )

    plan = hygiene.plan
    if plan is not None and plan.refs <= {e.id for e in answer.evidence}:
        try:
            verdict.value = plan.evaluate({e.id: e.value for e in answer.evidence})
        except PlanError as err:
            verdict.reasons.append(Reason("execution_error", f"{err.kind}: {err}"))
        verdict.reasons.extend(
            Reason(v.code, v.detail)
            for v in check_invariants(answer, plan, verdict.evidence_checks, doc)
        )

    if integrity:
        issues = integrity_issues if integrity_issues is not None else check_integrity(doc)
        rejects, warns = _integrity_reasons(answer, issues)
        verdict.reasons.extend(rejects)
        verdict.warnings.extend(warns)

    if not verdict.reasons:
        verdict.status = "accepted"
    return verdict
