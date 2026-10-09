import math

import pytest

from ledgermind.dsl import PlanError, compile_plan, execute

B = {"e1": 5829.0, "e2": 5735.0, "e3": 4000.0}


@pytest.mark.parametrize(
    "plan, expected",
    [
        ("(e1 - e2) / e2", (5829 - 5735) / 5735),
        ("(e1 - e2) / e2 * 100", (5829 - 5735) / 5735 * 100),
        ("pct_change(e2, e1)", (5829 - 5735) / 5735 * 100),
        ("sub(e1, e2)", 94.0),
        ("div(add(e1, e2), 2)", (5829 + 5735) / 2),
        ("mean(e1, e2, e3)", (5829 + 5735 + 4000) / 3),
        ("sum(e1, e2, e3)", 15564.0),
        ("max(e1, e2, e3) - min(e1, e2, e3)", 1829.0),
        ("e1 * 1000000", 5829e6),
        ("-e1 + e2", -94.0),
        ("pow(e3 / e2, 1 / 3) - 1", (4000 / 5735) ** (1 / 3) - 1),
    ],
)
def test_arithmetic(plan, expected):
    assert math.isclose(execute(plan, B), expected, rel_tol=1e-12)


def test_greater_returns_boolean():
    assert execute("greater(e1, e2)", B) is True
    assert compile_plan("greater(e1, e2)").returns_bool


def test_compile_reports_refs_literals_and_functions():
    p = compile_plan("mean(e1, e3) / e2 * 100")
    assert p.refs == {"e1", "e2", "e3"}
    assert p.literals == (100.0,)
    assert p.functions == {"mean"}
    assert not p.returns_bool


def test_compiled_plan_reevaluates_with_new_bindings():
    p = compile_plan("e1 - e2")
    assert p.evaluate({"e1": 3, "e2": 1}) == 2
    assert p.evaluate({"e1": 10, "e2": 1}) == 9


@pytest.mark.parametrize(
    "plan, kind",
    [
        ("1.64", "literal"),
        ("e1 * 0.5", "literal"),
        ("e1 + 42", "literal"),
        ("e1 + True", "literal"),
        ("e1 + 'x'", "literal"),
        ("__import__('os').system('ls')", "operation"),
        ("e1.real", "operation"),
        ("e1 ** 2", "operation"),
        ("e1 % e2", "operation"),
        ("e1 if e2 else e3", "operation"),
        ("[e1, e2]", "operation"),
        ("e1 > e2", "operation"),
        ("open(e1)", "operation"),
        ("x + e1", "unknown_ref"),
        ("e0 + e1", "unknown_ref"),
        ("div(e1)", "arity"),
        ("mean()", "arity"),
        ("max(*e1)", "arity"),
        ("greater(e1, e2) + 1", "type"),
        ("(e1 - ", "syntax"),
    ],
)
def test_rejections(plan, kind):
    with pytest.raises(PlanError) as info:
        execute(plan, B)
    assert info.value.kind == kind


def test_undefined_reference_at_evaluation():
    with pytest.raises(PlanError) as info:
        execute("e1 + e9", B)
    assert info.value.kind == "unknown_ref"


def test_math_errors():
    with pytest.raises(PlanError) as info:
        execute("e1 / (e2 - e2)", B)
    assert info.value.kind == "math"
    with pytest.raises(PlanError) as info:
        execute("pow(e1, e2)", B)
    assert info.value.kind == "math"


def test_complexity_limits():
    with pytest.raises(PlanError) as info:
        execute(" + ".join(["e1"] * 150), B)
    assert info.value.kind == "complexity"
    with pytest.raises(PlanError) as info:
        execute("(" * 40 + "e1" + ")" * 40 + " + " + "-" * 40 + "e1", B)
    assert info.value.kind == "complexity"
