import pytest

from ledgermind.eval.stats import bootstrap_ci, mcnemar, paired_bootstrap, percentile


def test_bootstrap_interval_contains_mean_and_has_sane_width():
    values = [1.0] * 700 + [0.0] * 300
    ci = bootstrap_ci(values)
    assert ci.mean == pytest.approx(0.7)
    assert ci.low < 0.7 < ci.high
    assert 0.04 < ci.high - ci.low < 0.08  # about plus or minus 2.8 points at n=1000


def test_bootstrap_is_deterministic_for_a_seed():
    values = [1.0, 0.0, 1.0, 1.0]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_paired_bootstrap_detects_a_real_difference():
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 60 + [0.0] * 40
    result = paired_bootstrap(a, b)
    assert result["diff"] == pytest.approx(0.2)
    assert result["low"] > 0 and result["p_value"] < 0.01


def test_paired_bootstrap_finds_no_difference_between_equal_systems():
    a = [1.0, 0.0] * 50
    assert paired_bootstrap(a, list(a))["diff"] == 0


def test_mcnemar():
    a = [True] * 30 + [False] * 10 + [True] * 50
    b = [False] * 30 + [True] * 10 + [True] * 50
    result = mcnemar(a, b)
    assert (result["only_a"], result["only_b"]) == (30, 10)
    assert result["p_value"] < 0.01
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0


def test_percentile():
    assert percentile([3, 1, 2, 4], 0.5) == pytest.approx(2.5)
