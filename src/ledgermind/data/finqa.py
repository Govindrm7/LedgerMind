"""FinQA loader (Chen et al., EMNLP 2021).

Files are fetched from the upstream repository at a pinned commit and verified by
SHA256, so every run sees byte identical data. The processed ``table`` field is used as
the canonical table because gold programs and ``gold_inds`` are aligned to it (the
original ``table_ori`` differs in shape for about 17% of examples).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ledgermind.document import Document

Split = Literal["train", "dev", "test"]

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/{commit}/dataset/{split}.json"
FINQA_SHA256: dict[str, str] = {
    "train": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dev": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}
DEFAULT_CACHE = Path("data/raw/finqa")


@dataclass(frozen=True)
class FinQAExample:
    id: str
    question: str
    document: Document
    program: str
    gold_answer: float | bool
    gold_inds: dict[str, str] = field(default_factory=dict)

    @property
    def is_boolean(self) -> bool:
        return isinstance(self.gold_answer, bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(split: Split, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Fetch a split into ``cache_dir`` (if missing) and verify its checksum."""
    path = Path(cache_dir) / f"{split}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = FINQA_URL.format(commit=FINQA_COMMIT, split=split)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (pinned https URL)
        tmp.rename(path)
    actual = _sha256(path)
    if actual != FINQA_SHA256[split]:
        raise ValueError(f"checksum mismatch for {path}: {actual}")
    return path


def _gold_answer(raw: float | str) -> float | bool:
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("yes", "no"):
            return lowered == "yes"
        return float(lowered)
    return float(raw)


def parse_example(raw: dict) -> FinQAExample:
    qa = raw["qa"]
    document = Document.build(
        doc_id=raw["filename"],
        table=raw["table"],
        sentences=list(raw["pre_text"]) + list(raw["post_text"]),
    )
    return FinQAExample(
        id=raw["id"],
        question=qa["question"].strip(),
        document=document,
        program=qa["program"],
        gold_answer=_gold_answer(qa["exe_ans"]),
        gold_inds=dict(qa.get("gold_inds", {})),
    )


def load_file(path: Path) -> list[FinQAExample]:
    with Path(path).open() as fh:
        return [parse_example(raw) for raw in json.load(fh)]


def load_split(split: Split, cache_dir: Path = DEFAULT_CACHE) -> list[FinQAExample]:
    return load_file(download(split, cache_dir))
