import json
from pathlib import Path

from ledgermind.data.convert import convert_all
from ledgermind.data.finqa import load_file
from ledgermind.data.prepare import write_jsonl
from ledgermind.eval.oracle import oracle_split

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


def test_oracle_on_fixture_is_exact(tmp_path):
    converted, report = convert_all(load_file(FIXTURE))
    path = tmp_path / "dev.jsonl"
    write_jsonl(converted, path)
    result = oracle_split(path, report.total)
    assert result["strict_accuracy_on_grounded"] == 1.0
    assert result["pipeline_ceiling_strict"] == 1.0
    assert result["mismatches"] == []
    json.dumps(result)
