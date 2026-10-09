import pytest

from ledgermind.numbers import extract_numbers
from ledgermind.saboteur.rewrite import flip_sign_cell, rewrite_value


@pytest.mark.parametrize(
    "text, new, expected",
    [
        ("$ 6427", 6500, "$ 6500"),
        ("$ -23158 ( 23158 )", -23000, "$ -23000 ( 23000 )"),
        ("12.5% ( 12.5 % )", 13.25, "13.2% ( 13.2 % )"),
        ("959.2", 1000.04, "1000.0"),
    ],
)
def test_rewrite_keeps_format_and_echo(text, new, expected):
    (m,) = extract_numbers(text)
    out = rewrite_value(text, m, new)
    assert out == expected
    assert len(extract_numbers(out)) == 1


def test_rewrite_one_of_several_numbers_in_a_sentence():
    text = "revenue was $ 3.8 million and costs were $ 1.9 million ."
    first, second = extract_numbers(text)
    out = rewrite_value(text, second, 2.4)
    assert [m.value for m in extract_numbers(out)] == [3.8, 2.4]


def test_rewrite_refuses_sign_change():
    (m,) = extract_numbers("$ 5")
    with pytest.raises(ValueError):
        rewrite_value("$ 5", m, -5)


@pytest.mark.parametrize(
    "text, value",
    [("$ 6427", -6427), ("$ -9457 ( 9457 )", 9457), ("12% ( 12 % )", -12), ("-4 ( 4 )", 4)],
)
def test_flip_sign_cell(text, value):
    (m,) = extract_numbers(flip_sign_cell(text))
    assert m.value == value


def test_flip_sign_cell_needs_one_nonzero_number():
    assert flip_sign_cell("0") is None
    assert flip_sign_cell("1 to 3") is None
