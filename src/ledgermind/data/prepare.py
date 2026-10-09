"""Build processed FinQA splits: grounded supervision targets plus a coverage report.

Usage::

    uv run python -m ledgermind.data.prepare --out data/processed/finqa

Writes ``{split}.jsonl`` (one converted example per line) and ``report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ledgermind.data.convert import ConvertedExample, convert_all
from ledgermind.data.finqa import DEFAULT_CACHE, load_split
from ledgermind.document import Document
from ledgermind.schema import Answer

SPLITS = ("train", "dev", "test")
DEFAULT_OUT = Path("data/processed/finqa")


def to_record(ex: ConvertedExample) -> dict:
    return {
        "id": ex.id,
        "question": ex.question,
        "document": {
            "doc_id": ex.document.doc_id,
            "table": [list(r) for r in ex.document.table],
            "sentences": list(ex.document.sentences),
        },
        "target": ex.target.model_dump(mode="json"),
        "gold_answer": ex.gold_answer,
        "flags": list(ex.flags),
    }


def from_record(record: dict) -> ConvertedExample:
    doc = record["document"]
    return ConvertedExample(
        id=record["id"],
        question=record["question"],
        document=Document.build(doc["doc_id"], doc["table"], doc["sentences"]),
        target=Answer.model_validate(record["target"]),
        gold_answer=record["gold_answer"],
        flags=tuple(record.get("flags", ())),
    )


def write_jsonl(examples: list[ConvertedExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(to_record(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[ConvertedExample]:
    with Path(path).open() as fh:
        return [from_record(json.loads(line)) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report: dict[str, dict] = {}
    for split in SPLITS:
        converted, split_report = convert_all(load_split(split, args.cache))
        write_jsonl(converted, args.out / f"{split}.jsonl")
        report[split] = split_report.as_dict()
        print(
            f"{split:>5}: {split_report.converted}/{split_report.total} grounded "
            f"({split_report.coverage:.2%}), failures {dict(split_report.failures)}"
        )

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
