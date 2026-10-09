from pathlib import Path

import pytest

from ledgermind.data import finqa
from ledgermind.schema import TableSource

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


@pytest.fixture(scope="module")
def examples():
    return {e.id: e for e in finqa.load_file(FIXTURE)}


def test_parses_all_fixture_examples(examples):
    assert len(examples) == 6
    ex = examples["ETR/2008/page_313.pdf-3"]
    assert ex.program == "subtract(959.2, 991.1), divide(#0, 991.1)"
    assert ex.gold_answer == pytest.approx(-0.03219)
    assert ex.document.doc_id == "ETR/2008/page_313.pdf"


def test_boolean_answers(examples):
    ex = examples["JPM/2018/page_73.pdf-5"]
    assert ex.is_boolean and ex.gold_answer is True


def test_sentences_concatenate_pre_and_post_text(examples):
    for ex in examples.values():
        for key, text in ex.gold_inds.items():
            kind, idx = key.split("_")
            if kind == "text":
                assert ex.document.sentences[int(idx)].strip() == text.strip()


def test_gold_table_rows_contain_program_numbers(examples):
    ex = examples["ETR/2008/page_313.pdf-3"]
    rows = [int(k.split("_")[1]) for k in ex.gold_inds if k.startswith("table")]
    values = {
        m.value
        for r in rows
        for c in range(len(ex.document.table[r]))
        for m in ex.document.numbers_at(TableSource(row=r, col=c))
    }
    assert {959.2, 991.1} <= values


def test_checksum_mismatch_is_detected(tmp_path):
    (tmp_path / "dev.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        finqa.download("dev", cache_dir=tmp_path)
