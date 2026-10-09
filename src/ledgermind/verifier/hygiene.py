"""Plan hygiene: the plan must be a real computation over the cited evidence.

Checks, in order:

* the plan compiles under the executor's whitelist and literal policy
* every referenced id is defined in the evidence list
* the result actually depends on the evidence (sensitivity check)
* the answer unit agrees with the plan's result type (boolean vs numeric)

The sensitivity check closes the hole the literal allowlist leaves open. ``e1 - e1 + 12``
only uses allowed literals and references evidence, yet its result never changes. The
check perturbs each referenced value in turn; if no perturbation moves the result, the
plan is a disguised constant and counts as a bypass.

Evidence that is cited but never used is reported as a warning, not a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ledgermind.dsl import CompiledPlan, PlanError, compile_plan
from ledgermind.schema import Answer

_PERTURBATIONS = (1.37, 0.61)


@dataclass
class HygieneResult:
    plan: CompiledPlan | None
    errors: list[tuple[str, str]] = field(default_factory=list)
    unused_evidence: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _depends_on_evidence(plan: CompiledPlan, bindings: dict[str, float]) -> bool:
    try:
        base = plan.evaluate(bindings)
    except PlanError:
        return True  # execution failures are reported separately
    for ref in plan.refs:
        for factor in _PERTURBATIONS:
            probe = dict(bindings)
            probe[ref] = bindings[ref] * factor + 1.0
            try:
                if plan.evaluate(probe) != base:
                    return True
            except PlanError:
                continue
    return False


def check_hygiene(answer: Answer) -> HygieneResult:
    try:
        plan = compile_plan(answer.plan)
    except PlanError as err:
        code = "bypass" if err.kind == "literal" else "plan_invalid"
        return HygieneResult(None, [(code, f"{err.kind}: {err}")])

    result = HygieneResult(plan)
    defined = {e.id for e in answer.evidence}
    undefined = sorted(plan.refs - defined)
    if undefined:
        result.errors.append(("unknown_ref", f"plan uses undefined evidence {undefined}"))
    if not plan.refs:
        result.errors.append(("bypass", "plan references no evidence"))
    result.unused_evidence = sorted(defined - plan.refs, key=lambda i: int(i[1:]))

    if plan.refs and not undefined:
        bindings = {e.id: e.value for e in answer.evidence}
        if not plan.returns_bool and not _depends_on_evidence(plan, bindings):
            result.errors.append(("bypass", "result does not depend on the cited evidence"))

    if plan.returns_bool != (answer.answer_unit == "boolean"):
        kind = "boolean" if plan.returns_bool else "numeric"
        result.errors.append(
            (
                "unit_mismatch",
                f"plan returns a {kind} result but answer_unit is {answer.answer_unit}",
            )
        )
    return result
