"""Document integrity: is the source internally consistent?

Provenance proves a figure is printed in the document; it cannot tell whether the
document itself was altered. These checks look for the fingerprints that tampering
leaves behind in financial tables.

**Footing.** For every row labelled ``total``, find the block of rows directly above it
that sums to the total (within printed rounding). Real tables foot in every amount column
with the same block. If the other columns foot with a block and one column does not, that
column has been altered, either in a line item or in the total itself.

**Text vs table agreement.** Narrative text often restates table figures. A number in a
sentence that differs from a table cell by exactly one digit, where the sentence names
the row (at least two content words, covering half the row label),
(``6179`` vs ``6178``) is flagged, unless the sentence value is also printed exactly
somewhere in the table.

Measured on clean FinQA dev and test documents, footing fires on about 1% and text
agreement on about 4%. Footing issues therefore reject an answer whose evidence they
touch; text disagreements are reported as warnings only.

Both checks are deliberately conservative: they only fire on positive evidence of an
inconsistency. Printed figures are rounded, so a total may differ from the sum of its
items by up to half a unit per item; a change smaller than that is invisible to footing.
A change to an isolated figure with no total and no restatement is undetectable from the
document alone, and the Saboteur reports it as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ledgermind.document import Document
from ledgermind.numbers import NumberMention, extract_numbers
from ledgermind.schema import Source, TableSource, TextSource

_TOTAL_RE = re.compile(r"^\s*(total|net total|grand total)\b|\btotal\s*$")
_NON_ADDITIVE_RE = re.compile(
    r"average|price|per share|rate|ratio|yield|percent|%|margin|may yet be|weighted"
)
_MONTH_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s*\.?\s*$"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d\d\b")
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOPWORDS = {"total", "with", "from", "that", "this", "were", "which", "year", "ended", "december"}


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    locations: tuple[Source, ...]
    detail: str
    severity: Literal["reject", "warn"] = "reject"


def _is_dash(text: str) -> bool:
    # FinQA renders an em dash (U+2014, "nil") in table cells as the digits 2014.
    return text.replace("$", "").strip() in ("2014", "-", "--", "")


def _cell_number(doc: Document, row: int, col: int) -> NumberMention | None:
    text = doc.table[row][col]
    if _is_dash(text):
        return NumberMention(0.0, False, 0, 0)
    mentions = doc.numbers_at(TableSource(row=row, col=col))
    if len(mentions) != 1 or mentions[0].is_percent:
        return None
    return mentions[0]


def _decimals(doc: Document, row: int, col: int) -> int:
    text = doc.table[row][col]
    m = re.search(r"\d\.(\d+)", text)
    return len(m.group(1)) if m else 0


def _footing_starts(doc: Document, total_row: int, col: int, subtotals: dict[int, int]) -> set[int]:
    """Start rows j such that rows j..total_row-1 sum to the total in column ``col``.

    A subtotal row found earlier counts as one item and its own block is skipped, so that
    nested totals (``total africa`` inside ``total``) foot correctly.
    """
    total = _cell_number(doc, total_row, col)
    if total is None:
        return set()
    starts: set[int] = set()
    running = 0.0
    rows = 0
    max_dec = _decimals(doc, total_row, col)
    j = total_row - 1
    while j >= 1:
        item = _cell_number(doc, j, col)
        if item is not None:
            running += item.value
            rows += 1
            max_dec = max(max_dec, _decimals(doc, j, col))
            tolerance = (rows + 1) * 0.5 * 10.0 ** (-max_dec) + 1e-9
            if rows >= 2 and abs(running - total.value) <= tolerance:
                starts.add(j)
        j = subtotals[j] - 1 if j in subtotals else j - 1
    return starts


def _additive_column(doc: Document, col: int) -> bool:
    return not _NON_ADDITIVE_RE.search(doc.column_header(col).lower())


def check_footings(doc: Document) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    subtotals: dict[int, int] = {}
    for row in range(1, doc.n_rows):
        if not _TOTAL_RE.search(doc.row_label(row).lower()):
            continue
        cols = [
            c
            for c in range(1, len(doc.table[row]))
            if _additive_column(doc, c)
            and not _is_dash(doc.table[row][c])
            and _cell_number(doc, row, c) is not None
        ]
        starts = {c: _footing_starts(doc, row, c, subtotals) for c in cols}
        footed = [c for c in cols if starts[c]]
        if footed:
            consensus = set.intersection(*(starts[c] for c in footed))
            block_start = max(consensus) if consensus else max(starts[footed[0]])
            subtotals[row] = block_start
            for c in cols:
                if starts[c]:
                    continue
                locations = tuple(TableSource(row=r, col=c) for r in range(block_start, row + 1))
                issues.append(
                    IntegrityIssue(
                        "footing_broken",
                        locations,
                        f"column {c} does not foot to '{doc.row_label(row)}' while "
                        f"{len(footed)} other column(s) do",
                    )
                )
    return issues


def _digit_shape(m: NumberMention, text: str) -> tuple[str, str]:
    """Integer and fractional digit strings, so 29.0 and 200 never compare as close."""
    raw = text[m.start : m.end].replace(",", "")
    whole, _, frac = raw.partition(".")
    return whole, frac


def _is_day_of_month(m: NumberMention, text: str) -> bool:
    return bool(_MONTH_RE.search(text[max(0, m.start - 12) : m.start]))


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _is_year(m: NumberMention) -> bool:
    return float(m.value).is_integer() and 1900 <= m.value <= 2100


def check_text_table_agreement(doc: Document) -> list[IntegrityIssue]:
    table_values: list[tuple[TableSource, NumberMention, str]] = []
    exact: set[float] = set()
    for r in range(1, doc.n_rows):
        for c in range(1, len(doc.table[r])):
            text = doc.table[r][c]
            if _is_dash(text):
                continue
            for m in extract_numbers(text):
                exact.add(m.value)
                if abs(m.value) >= 10 and not _is_year(m):
                    table_values.append((TableSource(row=r, col=c), m, _digit_shape(m, text)))

    issues: list[IntegrityIssue] = []
    for i, sentence in enumerate(doc.sentences):
        words = _content_words(sentence)
        for m in extract_numbers(sentence):
            if abs(m.value) < 10 or _is_year(m) or m.value in exact:
                continue
            if _is_day_of_month(m, sentence):
                continue
            shape = _digit_shape(m, sentence)
            for source, tm, tshape in table_values:
                if m.is_percent != tm.is_percent:
                    continue
                if tuple(map(len, shape)) != tuple(map(len, tshape)):
                    continue
                a, b = "".join(shape), "".join(tshape)
                if sum(x != y for x, y in zip(a, b, strict=True)) != 1:
                    continue
                label_words = _content_words(doc.row_label(source.row))
                shared = words & label_words
                if len(shared) < 2 or len(shared) < 0.5 * len(label_words):
                    continue
                years = set(_YEAR_RE.findall(doc.column_header(source.col)))
                if years and not years & set(_YEAR_RE.findall(sentence)):
                    continue
                issues.append(
                    IntegrityIssue(
                        "text_table_disagreement",
                        (TextSource(sent=i), source),
                        f"text says {sentence[m.start : m.end]}, table says "
                        f"{doc.table[source.row][source.col].strip()} for "
                        f"'{doc.row_label(source.row).strip()}'",
                        severity="warn",
                    )
                )
    return issues


def check_integrity(doc: Document) -> list[IntegrityIssue]:
    return check_footings(doc) + check_text_table_agreement(doc)
