"""Safe executor for computation plans.

A plan is a small arithmetic expression over evidence ids, for example
``(e1 - e2) / e2 * 100`` or ``mean(e1, e2, e3)``. It is parsed with :mod:`ast` and
evaluated by walking a whitelisted subset of the tree. Nothing is ever passed to
``eval``.

Two policies make "the model never writes the final number" enforceable:

* **Literal allowlist.** Only unit and scale constants (``100``, ``1000``, ``1e6`` and
  small integers used for averaging) may appear in a plan. A plan containing ``1.64``
  is rejected outright.
* **Evidence references.** The verifier additionally requires that the result depends
  on the cited evidence (see ``ledgermind.verifier``), which closes the remaining hole
  of plans like ``e1 - e1 + 12``.

Plans are compiled once and can be evaluated many times with different bindings, which
the verifier uses for its sensitivity check.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

# FinQA constants (const_m1, const_1 ... const_10, const_100 ... const_1000000000)
# plus 12 and 365 for month and day conversions.
ALLOWED_LITERALS: frozenset[float] = frozenset(
    [*range(1, 11), 12, 100, 365, 1_000, 10_000, 100_000, 1_000_000, 1_000_000_000]
)

MAX_PLAN_CHARS = 2000
MAX_NODES = 200
MAX_DEPTH = 24
MAX_ABS_EXPONENT = 100

_REF_RE = re.compile(r"^e[1-9][0-9]*$")

ErrorKind = Literal[
    "syntax", "unknown_ref", "literal", "operation", "arity", "type", "math", "complexity"
]

Value = float | bool


class PlanError(Exception):
    """A plan that cannot be compiled or executed. ``kind`` is stable for metrics."""

    def __init__(self, kind: ErrorKind, message: str):
        super().__init__(message)
        self.kind = kind


def _num(x: Value, where: str) -> float:
    if isinstance(x, bool):
        raise PlanError("type", f"{where} expects a number, got a boolean")
    return x


def _div(a: float, b: float) -> float:
    if b == 0:
        raise PlanError("math", "division by zero")
    return a / b


def _pow(a: float, b: float) -> float:
    if abs(b) > MAX_ABS_EXPONENT:
        raise PlanError("math", f"exponent {b} exceeds limit {MAX_ABS_EXPONENT}")
    try:
        return math.pow(a, b)
    except (OverflowError, ValueError) as exc:
        raise PlanError("math", f"pow failed: {exc}") from exc


def _pct_change(old: float, new: float) -> float:
    return _div(new - old, old) * 100.0


# name -> (callable, min args, max args or None for variadic, returns boolean)
_FUNCTIONS: dict[str, tuple[Callable[..., Value], int, int | None, bool]] = {
    "add": (lambda a, b: a + b, 2, 2, False),
    "sub": (lambda a, b: a - b, 2, 2, False),
    "mul": (lambda a, b: a * b, 2, 2, False),
    "div": (_div, 2, 2, False),
    "pow": (_pow, 2, 2, False),
    "pct_change": (_pct_change, 2, 2, False),
    "sum": (lambda *xs: math.fsum(xs), 1, None, False),
    "mean": (lambda *xs: math.fsum(xs) / len(xs), 1, None, False),
    "max": (lambda *xs: max(xs), 1, None, False),
    "min": (lambda *xs: min(xs), 1, None, False),
    "greater": (lambda a, b: a > b, 2, 2, True),
}

FUNCTION_NAMES = frozenset(_FUNCTIONS)

_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _div,
}


@dataclass(frozen=True)
class CompiledPlan:
    source: str
    tree: ast.expr
    refs: frozenset[str]
    literals: tuple[float, ...]
    functions: frozenset[str]
    returns_bool: bool

    def evaluate(self, bindings: Mapping[str, float]) -> Value:
        missing = self.refs - bindings.keys()
        if missing:
            raise PlanError("unknown_ref", f"plan references undefined evidence: {sorted(missing)}")
        result = _eval(self.tree, bindings)
        if not isinstance(result, bool) and not math.isfinite(result):
            raise PlanError("math", "result is not finite")
        return result


def compile_plan(plan: str) -> CompiledPlan:
    """Parse and validate a plan without evaluating it."""
    if len(plan) > MAX_PLAN_CHARS:
        raise PlanError("complexity", f"plan exceeds {MAX_PLAN_CHARS} characters")
    try:
        tree = ast.parse(plan.strip(), mode="eval").body
    except SyntaxError as exc:
        raise PlanError("syntax", f"invalid plan syntax: {exc.msg}") from exc

    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > MAX_NODES:
        raise PlanError("complexity", f"plan has {nodes} nodes, limit is {MAX_NODES}")

    refs: set[str] = set()
    literals: list[float] = []
    functions: set[str] = set()
    _validate(tree, refs, literals, functions, depth=0)
    returns_bool = isinstance(tree, ast.Call) and _FUNCTIONS[tree.func.id][3]  # type: ignore[union-attr]
    return CompiledPlan(
        plan, tree, frozenset(refs), tuple(literals), frozenset(functions), returns_bool
    )


def execute(plan: str, bindings: Mapping[str, float]) -> Value:
    """Compile and evaluate in one step."""
    return compile_plan(plan).evaluate(bindings)


def _validate(
    node: ast.AST, refs: set[str], literals: list[float], functions: set[str], depth: int
) -> None:
    if depth > MAX_DEPTH:
        raise PlanError("complexity", f"plan nesting exceeds depth {MAX_DEPTH}")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise PlanError("literal", f"non numeric literal {node.value!r}")
        if float(node.value) not in ALLOWED_LITERALS:
            raise PlanError("literal", f"literal {node.value!r} is not an allowed constant")
        literals.append(float(node.value))
    elif isinstance(node, ast.Name):
        if not _REF_RE.match(node.id):
            raise PlanError("unknown_ref", f"unknown name {node.id!r}")
        refs.add(node.id)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise PlanError("operation", f"operator {type(node.op).__name__} is not allowed")
        _validate(node.left, refs, literals, functions, depth + 1)
        _validate(node.right, refs, literals, functions, depth + 1)
        _reject_bool_operand(node.left)
        _reject_bool_operand(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise PlanError("operation", f"unary {type(node.op).__name__} is not allowed")
        _validate(node.operand, refs, literals, functions, depth + 1)
        _reject_bool_operand(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            name = getattr(node.func, "id", ast.unparse(node.func))
            raise PlanError("operation", f"function {name!r} is not allowed")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise PlanError("arity", "keyword and starred arguments are not allowed")
        _, lo, hi, _ = _FUNCTIONS[node.func.id]
        n = len(node.args)
        if n < lo or (hi is not None and n > hi):
            raise PlanError(
                "arity", f"{node.func.id} takes {lo if hi == lo else f'{lo}+'} args, got {n}"
            )
        functions.add(node.func.id)
        for arg in node.args:
            _validate(arg, refs, literals, functions, depth + 1)
            _reject_bool_operand(arg)
    else:
        raise PlanError("operation", f"{type(node).__name__} is not allowed in a plan")


def _reject_bool_operand(node: ast.AST) -> None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        spec = _FUNCTIONS.get(node.func.id)
        if spec and spec[3]:
            raise PlanError("type", f"{node.func.id} returns a boolean and cannot be an operand")


def _eval(node: ast.AST, bindings: Mapping[str, float]) -> Value:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return float(bindings[node.id])
    if isinstance(node, ast.BinOp):
        left = _num(_eval(node.left, bindings), "operator")
        right = _num(_eval(node.right, bindings), "operator")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        value = _num(_eval(node.operand, bindings), "unary operator")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call):
        fn = _FUNCTIONS[node.func.id][0]  # type: ignore[union-attr]
        args = [_num(_eval(a, bindings), node.func.id) for a in node.args]  # type: ignore[union-attr]
        return fn(*args)
    raise PlanError("operation", f"{type(node).__name__} is not allowed in a plan")
