"""Invariant checks: dimensional analysis over the plan.

Each evidence item is given a dimension read off the document:

* **kind**: ``percent`` if printed with ``%``, ``currency`` if printed with ``$``
* **scale**: a scale word next to the number, else in the row label, the column header,
  or the table's top left header cell (``( in millions )``), in that order
* **period**: ``quarter`` or ``annual`` when the column header says so

Dimensions propagate through the plan. Additive operations (``+ - sum mean max min
pct_change greater``) require compatible operands: a percentage added to a dollar
amount, millions added to billions, or a quarterly figure added to an annual one is a
violation. Unknown attributes are compatible with anything, so the check only fires on
positive evidence of a mismatch. Multiplication and division change dimensions, so their
result is unknown, except that dividing a percent by 100 or applying a scale constant
clears the corresponding attribute.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace

from ledgermind.document import Document
from ledgermind.dsl import CompiledPlan
from ledgermind.schema import Answer, TableSource
from ledgermind.verifier.provenance import EvidenceCheck

_SCALE_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b|\b(thousand|million|billion)s?\b")
_QUARTER_RE = re.compile(r"\bq[1-4]\b|\bquarter\b|\bthree months\b|\b1st|\b2nd|\b3rd|\b4th")
_ANNUAL_RE = re.compile(
    r"^\s*(fiscal\s+)?(19|20)\d\d\s*$|\byear ended\b|\btwelve months\b|\bannual\b"
)
_ADDITIVE_FUNCS = {"add", "sub", "sum", "mean", "max", "min", "pct_change", "greater"}
_SCALE_LITERALS = {1_000.0, 1_000_000.0, 1_000_000_000.0}


@dataclass(frozen=True)
class Dim:
    kind: str | None = None
    scale: str | None = None
    period: str | None = None


UNKNOWN = Dim()


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str


def _scale_word(text: str) -> str | None:
    m = _SCALE_RE.search(text.lower())
    if not m:
        return None
    word = (m.group(1) or m.group(2)).rstrip("s")
    return word


def _period(header: str) -> str | None:
    h = header.lower()
    if _QUARTER_RE.search(h):
        return "quarter"
    if _ANNUAL_RE.search(h):
        return "annual"
    return None


def evidence_dim(check: EvidenceCheck, source, doc: Document) -> Dim:
    """Dimension of one verified evidence item. Unverified items are unknown."""
    if not check.ok or check.mention is None or check.cited_text is None:
        return UNKNOWN
    kind = (
        "percent" if check.mention.is_percent else "currency" if "$" in check.cited_text else None
    )
    scale = check.mention.scale_word
    period = None
    if isinstance(source, TableSource):
        header = doc.column_header(source.col)
        scale = (
            scale
            or _scale_word(doc.row_label(source.row))
            or _scale_word(header)
            or _scale_word(doc.column_header(0))  # table wide unit, e.g. "( in millions )"
        )
        period = _period(header)
    return Dim(kind, scale, period)


class _Checker:
    def __init__(self, dims: dict[str, Dim]):
        self.dims = dims
        self.violations: list[Violation] = []

    def merge(self, a: Dim, b: Dim, where: str) -> Dim:
        for attr, code in (
            ("kind", "unit_mismatch"),
            ("scale", "scale_mismatch"),
            ("period", "period_mismatch"),
        ):
            x, y = getattr(a, attr), getattr(b, attr)
            if x is not None and y is not None and x != y:
                self.violations.append(Violation(code, f"{x} combined with {y} in {where}"))
        return Dim(a.kind or b.kind, a.scale or b.scale, a.period or b.period)

    def scaled(self, dim: Dim, other: ast.AST, op: type) -> Dim:
        if not isinstance(other, ast.Constant):
            return UNKNOWN
        value = float(other.value)
        if op is ast.Div and value == 100 and dim.kind == "percent":
            return replace(dim, kind=None)
        if value in _SCALE_LITERALS:
            return replace(dim, scale=None)
        return dim

    def visit(self, node: ast.AST) -> Dim:
        if isinstance(node, ast.Name):
            return self.dims.get(node.id, UNKNOWN)
        if isinstance(node, ast.Constant):
            return UNKNOWN
        if isinstance(node, ast.UnaryOp):
            return self.visit(node.operand)
        where = ast.unparse(node)
        if isinstance(node, ast.BinOp):
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return self.merge(left, right, where)
            if isinstance(node.right, ast.Constant):
                return self.scaled(left, node.right, type(node.op))
            if isinstance(node.left, ast.Constant) and isinstance(node.op, ast.Mult):
                return self.scaled(right, node.left, ast.Mult)
            return UNKNOWN
        if isinstance(node, ast.Call):
            dims = [self.visit(a) for a in node.args]
            name = node.func.id  # type: ignore[union-attr]
            if name in _ADDITIVE_FUNCS:
                acc = dims[0]
                for d in dims[1:]:
                    acc = self.merge(acc, d, where)
                return UNKNOWN if name in ("pct_change", "greater") else acc
            return UNKNOWN
        return UNKNOWN


def check_invariants(
    answer: Answer, plan: CompiledPlan, checks: list[EvidenceCheck], doc: Document
) -> list[Violation]:
    by_id = {c.evidence_id: c for c in checks}
    dims = {e.id: evidence_dim(by_id[e.id], e.source, doc) for e in answer.evidence}
    checker = _Checker(dims)
    checker.visit(plan.tree)
    return checker.violations
