import pytest

from ledgermind.eval.answer import answers_match, answers_match_stated


@pytest.mark.parametrize(
    "pred, gold, strict, scale",
    [
        (-0.0321853, -0.03219, True, True),
        (108288.888888, 108288.88889, True, True),
        (1.62, 0.0162, False, True),
        (0.0162, 1.62, False, True),
        (0.0163, 0.0162, False, False),
        (5829.0, 5829.0, True, True),
        (5830.0, 5829.0, False, False),
        (0.0, 0.0, True, True),
        (None, 1.0, False, False),
    ],
)
def test_numeric_matching(pred, gold, strict, scale):
    assert answers_match(pred, gold, "strict") is strict
    assert answers_match(pred, gold, "scale") is scale


def test_boolean_matching():
    assert answers_match(True, True)
    assert not answers_match(False, True)
    assert not answers_match(1.0, True)
    assert not answers_match(True, 1.0)


def test_unknown_mode():
    with pytest.raises(ValueError):
        answers_match(1.0, 1.0, "fuzzy")


@pytest.mark.parametrize(
    "pred, decimals, gold, mode, expected",
    [
        (1.64, 2, 1.63657, "strict", True),  # gold rounded to the two decimals stated
        (27.81, 2, 27.80639, "strict", True),
        (7.2, 1, 0.07157, "scale", True),  # 7.2% stated for a fraction gold
        (7.2, 1, 0.07157, "strict", False),
        (13.2, 1, 0.12027, "scale", False),  # a genuinely wrong answer stays wrong
        (7.0, 0, 7.16, "strict", False),  # one significant digit is too coarse
        (0.35030674846625767, None, 0.3510, "strict", False),  # computed: ordinary match
        (0.35030674846625767, None, 0.35031, "strict", True),
        (None, 2, 1.0, "strict", False),
        (True, None, True, "strict", True),
    ],
)
def test_answers_match_stated(pred, decimals, gold, mode, expected):
    assert answers_match_stated(pred, decimals, gold, mode) is expected
