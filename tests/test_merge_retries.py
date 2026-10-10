import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("merge_retries", Path("scripts/merge_retries.py"))
merge_retries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_retries)


def test_only_failed_rows_are_replaced(tmp_path):
    base = [
        {"id": "a", "completion": "ok", "error": None},
        {"id": "b", "completion": "", "error": "ReadTimeout"},
        {"id": "c", "completion": "", "error": "ReadTimeout"},
    ]
    retry = [
        {"id": "a", "completion": "should not replace", "error": None},
        {"id": "b", "completion": "fixed", "error": None},
        {"id": "c", "completion": "", "error": "ReadTimeout"},
    ]
    for name, rows in (("base", base), ("retry", retry)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "merged.jsonl"
    merge_retries.main(
        [
            "--base",
            str(tmp_path / "base.jsonl"),
            "--retry",
            str(tmp_path / "retry.jsonl"),
            "--out",
            str(out),
        ]
    )
    merged = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["completion"] for r in merged] == ["ok", "fixed", ""]
    assert merged[2]["error"] == "ReadTimeout"
