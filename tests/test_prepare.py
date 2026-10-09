from pathlib import Path

from ledgermind.data.convert import convert_all
from ledgermind.data.finqa import load_file
from ledgermind.data.prepare import read_jsonl, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


def test_jsonl_round_trip(tmp_path):
    converted, _ = convert_all(load_file(FIXTURE))
    path = tmp_path / "sample.jsonl"
    write_jsonl(converted, path)
    restored = read_jsonl(path)
    assert restored == converted
