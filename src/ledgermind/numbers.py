"""Number extraction and normalization for financial text and table cells.

The provenance check depends on this module: a cited value is only accepted if it
matches a number that is actually printed at the cited location. Getting the sign
right matters most, because a flipped sign is one of the faults the Saboteur injects.

Formats handled (all observed in FinQA):

* plain and grouped numbers: ``6427``, ``23,158``, ``.9``
* currency: ``$ 6427``, ``$6,427``
* explicit negatives: ``-23158``, ``$ -9457``, ``- 9%`` at the start of a cell
* accounting negatives: ``(23,158)``, ``( $ 67 )``, ``( 9 ) %``
* FinQA echo annotations, where the processed text repeats a value in parentheses:
  ``$ -23158 ( 23158 )`` and ``2.05% ( 2.05 % )`` are each a single number
* scale words: ``$ 2.1 billion`` keeps value 2.1 and records the scale word

Values are kept exactly as printed. Scale words and percent signs are recorded as
attributes and never applied implicitly; any conversion belongs in the plan.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)")
_SCALE_RE = re.compile(r"\s*(thousand|million|billion|trillion)s?\b", re.IGNORECASE)
_MINUS = "-−"

SCALE_FACTORS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


@dataclass(frozen=True)
class NumberMention:
    """A number printed in a piece of text, with its location."""

    value: float
    is_percent: bool
    start: int
    end: int
    scale_word: str | None = None

    @property
    def magnitude(self) -> float:
        return abs(self.value)


@dataclass
class _Token:
    digits: str
    start: int
    end: int
    negative: bool = False
    is_percent: bool = False
    scale_word: str | None = None
    echo: bool = False

    @property
    def magnitude(self) -> float:
        return float(self.digits.replace(",", ""))


def _skip_spaces_back(text: str, i: int) -> int:
    while i >= 0 and text[i].isspace():
        i -= 1
    return i


def _skip_spaces_fwd(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _has_explicit_minus(text: str, start: int) -> bool:
    """True if the number starting at ``start`` carries a leading minus sign.

    A minus counts when it is attached to the number (``-5``, ``$ -5``, ``-$5``) or when
    it opens the string (``- 9%``). A spaced minus between two numbers (``2010 - 2012``)
    is treated as a range separator, not a sign.
    """
    j = _skip_spaces_back(text, start - 1)
    if j >= 0 and text[j] == "$":
        j = _skip_spaces_back(text, j - 1)
    if j < 0 or text[j] not in _MINUS:
        return False
    if not text[:j].strip():
        return True
    attached = text[j + 1 : start] in ("", "$")
    return attached and text[j - 1] in " ($"


def _paren_open_before(text: str, start: int) -> int | None:
    """Index of a ``(`` that opens directly before the number, allowing ``$``, a sign and spaces."""
    j = _skip_spaces_back(text, start - 1)
    if j >= 0 and text[j] in "+" + _MINUS:
        j = _skip_spaces_back(text, j - 1)
    if j >= 0 and text[j] == "$":
        j = _skip_spaces_back(text, j - 1)
    if j >= 0 and text[j] == "(":
        return j
    return None


def _scan_tokens(text: str) -> list[_Token]:
    return [_Token(m.group(1), m.start(1), m.end(1)) for m in _NUM_RE.finditer(text)]


def _mark_echoes(text: str, tokens: list[_Token]) -> None:
    """Flag FinQA echo annotations such as the ``( 23158 )`` in ``-23158 ( 23158 )``."""
    prev: _Token | None = None
    for tok in tokens:
        if prev is not None and not prev.echo:
            open_idx = _paren_open_before(text, tok.start)
            if open_idx is not None:
                gap = text[prev.end : open_idx]
                close_idx = _skip_spaces_fwd(text, tok.end)
                if close_idx < len(text) and text[close_idx] == "%":
                    close_idx = _skip_spaces_fwd(text, close_idx + 1)
                closes = close_idx < len(text) and text[close_idx] == ")"
                gap_ok = not gap.replace("%", "").strip()
                if closes and gap_ok and math.isclose(prev.magnitude, tok.magnitude):
                    tok.echo = True
                    tok.end = close_idx + 1
                    continue
        prev = tok


def _end_of_echo(tokens: list[_Token], idx: int) -> int | None:
    """If the token after ``idx`` is an echo, return the index just past its ``)``."""
    if idx + 1 < len(tokens) and tokens[idx + 1].echo:
        return tokens[idx + 1].end
    return None


def extract_numbers(text: str) -> list[NumberMention]:
    """Return every number printed in ``text`` with sign, percent flag and position."""
    tokens = _scan_tokens(text)
    _mark_echoes(text, tokens)

    mentions: list[NumberMention] = []
    for idx, tok in enumerate(tokens):
        if tok.echo:
            continue

        negative = _has_explicit_minus(text, tok.start)

        # Look past the number: optional %, optional echo group, optional closing paren.
        k = _skip_spaces_fwd(text, tok.end)
        is_percent = k < len(text) and text[k] == "%"
        if is_percent:
            k = _skip_spaces_fwd(text, k + 1)
        echo_end = _end_of_echo(tokens, idx)
        if echo_end is not None:
            k = _skip_spaces_fwd(text, echo_end)

        open_idx = _paren_open_before(text, tok.start)
        if open_idx is not None and k < len(text) and text[k] == ")":
            negative = True
            after = _skip_spaces_fwd(text, k + 1)
            if after < len(text) and text[after] == "%":
                is_percent = True
                k = after

        scale_word = None
        scale_match = _SCALE_RE.match(text, tok.end if not is_percent else k)
        if scale_match:
            scale_word = scale_match.group(1).lower()

        value = tok.magnitude * (-1.0 if negative else 1.0)
        mentions.append(NumberMention(value, is_percent, tok.start, tok.end, scale_word))
    return mentions


def parse_scalar(text: str) -> NumberMention | None:
    """Parse a string that should contain exactly one number, such as a program argument."""
    mentions = extract_numbers(text.strip())
    if len(mentions) != 1:
        return None
    return mentions[0]


def numbers_equal(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Exact comparison up to float representation. Used for provenance, not for answers."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
