import json

import pytest

from ledgermind.data.sample import main, sample_ids


def test_sample_is_seeded_and_keeps_original_order():
    ids = [f"q{i}" for i in range(100)]
    a, b = sample_ids(ids, 10, seed=0), sample_ids(ids, 10, seed=0)
    assert a == b and len(set(a)) == 10
    assert a == sorted(a, key=ids.index)
    assert sample_ids(ids, 10, seed=1) != a


def test_sample_rejects_more_than_available():
    with pytest.raises(ValueError, match="cannot sample"):
        sample_ids(["a", "b"], 3, seed=0)


def test_main_writes_subset_and_ids(tmp_path):
    rows = [{"id": f"q{i}", "question": "x"} for i in range(20)]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids_out = tmp_path / "results" / "ids.json"
    chosen = main(["--data", str(tmp_path), "--n", "5", "--ids-out", str(ids_out)])
    subset = [
        json.loads(line) for line in (tmp_path / "test_sample5.jsonl").read_text().splitlines()
    ]
    assert [r["id"] for r in subset] == chosen and len(chosen) == 5
    assert json.loads(ids_out.read_text())["ids"] == chosen
