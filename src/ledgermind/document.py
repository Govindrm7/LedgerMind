"""Source documents and the addressing scheme that evidence pointers resolve against.

A document is a table (row 0 is the header, column 0 holds row labels) plus a list of
sentences. Documents are immutable; the Saboteur derives corrupted copies through the
``with_*`` methods so that the original is never modified in place.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from ledgermind.numbers import NumberMention, extract_numbers, numbers_equal
from ledgermind.schema import Source, TableSource, TextSource


@dataclass(frozen=True)
class Document:
    doc_id: str
    table: tuple[tuple[str, ...], ...]
    sentences: tuple[str, ...]

    @classmethod
    def build(cls, doc_id: str, table: list[list[str]], sentences: list[str]) -> Document:
        return cls(doc_id, tuple(tuple(r) for r in table), tuple(sentences))

    @property
    def n_rows(self) -> int:
        return len(self.table)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.table), default=0)

    def resolve(self, source: Source) -> str | None:
        """Text at the addressed location, or None if the pointer is out of range."""
        if isinstance(source, TableSource):
            if source.row < len(self.table) and source.col < len(self.table[source.row]):
                return self.table[source.row][source.col]
            return None
        if isinstance(source, TextSource):
            if source.sent < len(self.sentences):
                return self.sentences[source.sent]
            return None
        raise TypeError(f"unknown source type: {type(source).__name__}")

    def numbers_at(self, source: Source) -> list[NumberMention]:
        text = self.resolve(source)
        return [] if text is None else extract_numbers(text)

    def locations(self) -> Iterator[Source]:
        """Every addressable location, table cells first."""
        for r, row in enumerate(self.table):
            for c in range(len(row)):
                yield TableSource(row=r, col=c)
        for i in range(len(self.sentences)):
            yield TextSource(sent=i)

    def find_value(self, value: float) -> list[Source]:
        """All locations where ``value`` is printed."""
        return [
            loc
            for loc in self.locations()
            if any(numbers_equal(m.value, value) for m in self.numbers_at(loc))
        ]

    def row_label(self, row: int) -> str:
        return self.table[row][0] if row < len(self.table) and self.table[row] else ""

    def column_header(self, col: int) -> str:
        return self.table[0][col] if self.table and col < len(self.table[0]) else ""

    def with_cell(self, row: int, col: int, text: str) -> Document:
        rows = [list(r) for r in self.table]
        rows[row][col] = text
        return replace(self, table=tuple(tuple(r) for r in rows))

    def with_sentence(self, index: int, text: str) -> Document:
        sents = list(self.sentences)
        sents[index] = text
        return replace(self, sentences=tuple(sents))

    def without_row(self, row: int) -> Document:
        return replace(self, table=self.table[:row] + self.table[row + 1 :])

    def render(self) -> str:
        """Render with explicit coordinates so that the model can cite them."""
        lines: list[str] = []
        if self.table:
            lines.append("TABLE (cite as row, col; row 0 is the header)")
            header = " | ".join(f"c{c}" for c in range(self.n_cols))
            lines.append(f"r | {header}")
            for r, row in enumerate(self.table):
                cells = " | ".join(cell.replace("|", "/").strip() for cell in row)
                lines.append(f"r{r} | {cells}")
        if self.sentences:
            if lines:
                lines.append("")
            lines.append("TEXT (cite as sent)")
            lines.extend(f"[s{i}] {s.strip()}" for i, s in enumerate(self.sentences))
        return "\n".join(lines)
