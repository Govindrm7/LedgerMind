import pytest

from ledgermind.numbers import extract_numbers, numbers_equal, parse_scalar


def values(text: str) -> list[float]:
    return [m.value for m in extract_numbers(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("6427", [6427]),
        ("$ 6427", [6427]),
        ("$6,427", [6427]),
        ("23,158.5", [23158.5]),
        (".9", [0.9]),
        ("$ 9.9 billion", [9.9]),
    ],
)
def test_plain_and_currency(text, expected):
    assert values(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("-23158", [-23158]),
        ("$ -9457", [-9457]),
        ("-$5", [-5]),
        ("- 9% ( 9 % )", [-9]),
        ("(23,158)", [-23158]),
        ("$(23,158)", [-23158]),
        ("( $ 67 )", [-67]),
        ("( 48 )", [-48]),
    ],
)
def test_negatives(text, expected):
    assert values(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$ -23158 ( 23158 )", [-23158]),
        ("-6781 ( 6781 )", [-6781]),
        ("-.9 ( .9 )", [-0.9]),
        ("2.05% ( 2.05 % )", [2.05]),
        ("+9% ( +9 % )", [9]),
    ],
)
def test_finqa_echo_annotations_collapse(text, expected):
    assert values(text) == expected


def test_percent_flags():
    (m,) = extract_numbers("12.5% ( 12.5 % )")
    assert m.is_percent and m.value == 12.5
    (m,) = extract_numbers("( 9 ) % (  % )")
    assert m.is_percent and m.value == -9
    (m,) = extract_numbers("$ 6427")
    assert not m.is_percent


def test_nested_accounting_percent_with_echo():
    (m,) = extract_numbers("( 9.9% ( 9.9 % ) )")
    assert m.value == -9.9 and m.is_percent


def test_ranges_are_not_negative():
    assert values("2010 - 2012") == [2010, 2012]
    assert all(v > 0 for v in values("2.0%-3.5% ( 2.0%-3.5 % ) above libor"))


def test_digits_inside_words_are_ignored():
    assert values("q4 results in form 10-k") == [10]


def test_sentence_with_several_numbers():
    text = (
        "if libor changes by 100 basis points , "
        "our annual interest expense would change by $ 3.8 million ."
    )
    mentions = extract_numbers(text)
    assert [m.value for m in mentions] == [100, 3.8]
    assert mentions[1].scale_word == "million"
    assert text[mentions[1].start : mentions[1].end] == "3.8"


def test_parse_scalar():
    assert parse_scalar("-9999").value == -9999
    assert parse_scalar("9.9%").is_percent
    assert parse_scalar("operating profit") is None
    assert parse_scalar("1 2") is None


def test_numbers_equal():
    assert numbers_equal(0.1 + 0.2, 0.3)
    assert not numbers_equal(6427, 6428)
    assert not numbers_equal(-9457, 9457)
