"""Rewrite numbers inside cells and sentences while keeping their printed format.

FinQA's processed text repeats some values in an echo group (``-23158 ( 23158 )``,
``12% ( 12 % )``). A rewrite has to change the echo too, otherwise the corrupted text
would carry two different numbers and be trivially detectable.
"""

from __future__ import annotations

import re

from ledgermind.numbers import NumberMention, extract_numbers


def format_magnitude(value: float, decimals: int) -> str:
    return f"{abs(value):.{decimals}f}"


def decimals_of(text: str, mention: NumberMention) -> int:
    token = text[mention.start : mention.end]
    return len(token.split(".")[1]) if "." in token else 0


def replace_mention(text: str, mention: NumberMention, new_magnitude: str) -> str:
    """Replace the digits of ``mention`` and of its echo group, keeping sign, ``$`` and ``%``."""
    old = text[mention.start : mention.end]
    head, tail = text[: mention.start], text[mention.end :]
    echo = re.compile(r"^(\s*%?\s*\(\s*[-+]?\s*)" + re.escape(old) + r"(\s*%?\s*\))")
    tail = echo.sub(lambda m: m.group(1) + new_magnitude + m.group(2), tail, count=1)
    return head + new_magnitude + tail


def rewrite_value(text: str, mention: NumberMention, new_value: float) -> str:
    """Set the printed value to ``new_value``; the sign must match the original."""
    if (new_value < 0) != (mention.value < 0) and new_value != 0:
        raise ValueError("rewrite_value keeps the sign; use flip_sign_cell to change it")
    return replace_mention(text, mention, format_magnitude(new_value, decimals_of(text, mention)))


def flip_sign_cell(text: str) -> str | None:
    """Flip the sign of a single number table cell in FinQA's processed style."""
    mentions = extract_numbers(text)
    if len(mentions) != 1 or mentions[0].value == 0:
        return None
    m = mentions[0]
    digits = text[m.start : m.end]
    prefix = "$ " if "$" in text else ""
    pct = "%" if m.is_percent else ""
    if m.value > 0:
        echo = f" ( {digits} % )" if pct else f" ( {digits} )"
        return f"{prefix}-{digits}{pct}{echo}"
    return f"{prefix}{digits}{pct}"
