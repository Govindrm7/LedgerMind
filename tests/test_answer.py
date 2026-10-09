import pytest

from ledgermind.eval.answer import answers_match


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
