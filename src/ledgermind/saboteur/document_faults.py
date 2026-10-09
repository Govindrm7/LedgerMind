"""Saboteur Mode B: corrupt the source document.

Two questions are asked of every corrupted document:

* **Stale answer.** An answer computed from the original document is checked against the
  corrupted one. Provenance must reject it, because a cited figure is no longer printed
  where it is cited. This is the "the filing changed after we answered" scenario.
* **Faithful reader.** A model that reads the corrupted document correctly cites the
  corrupted figures, so provenance passes, as it should. Only document integrity checks
  can notice the tampering, and only where the document has internal redundancy
  (a total that foots). We report detection on all cases and on the detectable subset.

Faults: ``digit_change`` (one digit of a cited table cell), ``sign_flip``, ``swap_columns``
(a cited cell swapped with another column of the same row, the classic quarter versus
full year mixup) and ``delete_row`` (the cited line item is removed).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ledgermind.document import Document
from ledgermind.numbers import numbers_equal
from ledgermind.saboteur.rewrite import flip_sign_cell, replace_mention
from ledgermind.schema import Answer, TableSource


@dataclass(frozen=True)
class DocumentFault:
    fault: str
    document: Document
    cell: TableSource
    faithful: Answer | None


def _table_evidence(answer: Answer) -> list:
    return [e for e in answer.evidence if isinstance(e.source, TableSource) and e.source.row > 0]


def _faithful(answer: Answer, doc: Document) -> Answer | None:
    """The answer a correct reader of ``doc`` would give with the same pointers."""
    evidence = []
    for e in answer.evidence:
        mentions = doc.numbers_at(e.source)
        if not mentions:
            return None
        same = [m for m in mentions if numbers_equal(m.value, e.value)]
        value = same[0].value if same else mentions[0].value
        evidence.append(e.model_copy(update={"value": value}))
    return answer.model_copy(update={"evidence": evidence})


def _single_number(doc: Document, cell: TableSource):
    mentions = doc.numbers_at(cell)
    return mentions[0] if len(mentions) == 1 else None


def doc_digit_change(answer: Answer, doc: Document, rng: random.Random) -> DocumentFault | None:
    options = [e for e in _table_evidence(answer) if _single_number(doc, e.source)]
    if not options:
        return None
    ev = rng.choice(options)
    text = doc.table[ev.source.row][ev.source.col]
    m = _single_number(doc, ev.source)
    token = text[m.start : m.end]
    positions = [i for i, ch in enumerate(token) if ch.isdigit()]
    i = rng.choice(positions)
    digits = [d for d in "0123456789" if d != token[i] and not (i == 0 and d == "0")]
    new_token = token[:i] + rng.choice(digits) + token[i + 1 :]
    corrupted = doc.with_cell(ev.source.row, ev.source.col, replace_mention(text, m, new_token))
    return DocumentFault("digit_change", corrupted, ev.source, _faithful(answer, corrupted))


def doc_sign_flip(answer: Answer, doc: Document, rng: random.Random) -> DocumentFault | None:
    options = [
        e
        for e in _table_evidence(answer)
        if flip_sign_cell(doc.table[e.source.row][e.source.col]) is not None
    ]
    if not options:
        return None
    ev = rng.choice(options)
    flipped = flip_sign_cell(doc.table[ev.source.row][ev.source.col])
    corrupted = doc.with_cell(ev.source.row, ev.source.col, flipped)
    return DocumentFault("sign_flip", corrupted, ev.source, _faithful(answer, corrupted))


def swap_columns(answer: Answer, doc: Document, rng: random.Random) -> DocumentFault | None:
    options = []
    for e in _table_evidence(answer):
        row = doc.table[e.source.row]
        mine = _single_number(doc, e.source)
        if mine is None:
            continue
        for c in range(1, len(row)):
            other = _single_number(doc, TableSource(row=e.source.row, col=c))
            if (
                c != e.source.col
                and other is not None
                and not numbers_equal(other.value, mine.value)
            ):
                options.append((e, c))
    if not options:
        return None
    ev, col = rng.choice(options)
    r = ev.source.row
    a, b = doc.table[r][ev.source.col], doc.table[r][col]
    corrupted = doc.with_cell(r, ev.source.col, b).with_cell(r, col, a)
    return DocumentFault("swap_columns", corrupted, ev.source, _faithful(answer, corrupted))


def delete_row(answer: Answer, doc: Document, rng: random.Random) -> DocumentFault | None:
    options = _table_evidence(answer)
    if not options:
        return None
    ev = rng.choice(options)
    corrupted = doc.without_row(ev.source.row)
    return DocumentFault("delete_row", corrupted, ev.source, None)


DOCUMENT_FAULTS: dict[str, Callable[[Answer, Document, random.Random], DocumentFault | None]] = {
    "digit_change": doc_digit_change,
    "sign_flip": doc_sign_flip,
    "swap_columns": swap_columns,
    "delete_row": delete_row,
}
