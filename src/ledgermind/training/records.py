"""Dataset records for SFT and GRPO.

SFT uses TRL's prompt and completion format, so the loss covers only the JSON target.
GRPO records carry the document and gold answer as JSON strings: Arrow cannot store a
column that mixes floats and booleans, and ragged tables are awkward as nested types.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ledgermind.data.convert import ConvertedExample
from ledgermind.data.prompts import build_prompt
from ledgermind.document import Document
from ledgermind.schema import to_json


def sft_record(ex: ConvertedExample) -> dict:
    return {
        "id": ex.id,
        "prompt": build_prompt(ex.question, ex.document),
        "completion": to_json(ex.target),
    }


def document_json(doc: Document) -> str:
    return json.dumps(
        {
            "doc_id": doc.doc_id,
            "table": [list(r) for r in doc.table],
            "sentences": list(doc.sentences),
        }
    )


@lru_cache(maxsize=4096)
def document_from_json(payload: str) -> Document:
    d = json.loads(payload)
    return Document.build(d["doc_id"], d["table"], d["sentences"])


def grpo_record(ex: ConvertedExample) -> dict:
    return {
        "id": ex.id,
        "prompt": build_prompt(ex.question, ex.document),
        "document_json": document_json(ex.document),
        "gold_json": json.dumps(ex.gold_answer),
    }
