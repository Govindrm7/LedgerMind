"""Provenance gate: every cited figure must be printed at the cited location.

Each evidence item gets one of four statuses:

* ``ok``: the value is printed at the cited cell or sentence
* ``misattributed``: the value is printed in the document, but not where it is cited
* ``fabricated``: the value is printed nowhere in the document
* ``bad_pointer``: the cited location does not exist

Only ``ok`` passes. Values are compared exactly as printed, including sign, so citing
23158 for a cell that reads ``-23158`` fails; the plan must negate explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ledgermind.document import Document
from ledgermind.numbers import NumberMention, numbers_equal
from ledgermind.schema import Answer, Evidence, Source

ProvenanceStatus = Literal["ok", "misattributed", "fabricated", "bad_pointer"]


@dataclass(frozen=True)
class EvidenceCheck:
    evidence_id: str
    status: ProvenanceStatus
    cited_text: str | None
    mention: NumberMention | None = None
    found_at: tuple[Source, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def check_evidence(evidence: Evidence, doc: Document) -> EvidenceCheck:
    cited = doc.resolve(evidence.source)
    if cited is None:
        return EvidenceCheck(evidence.id, "bad_pointer", None)
    for mention in doc.numbers_at(evidence.source):
        if numbers_equal(mention.value, evidence.value):
            return EvidenceCheck(evidence.id, "ok", cited, mention)
    elsewhere = tuple(doc.find_value(evidence.value))
    status: ProvenanceStatus = "misattributed" if elsewhere else "fabricated"
    return EvidenceCheck(evidence.id, status, cited, None, elsewhere)


def check_provenance(answer: Answer, doc: Document) -> list[EvidenceCheck]:
    return [check_evidence(e, doc) for e in answer.evidence]
