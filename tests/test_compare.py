import json

import pytest

from ledgermind.eval.compare import compare, main


def rows(flags):
    return {
        f"q{i}": {"id": f"q{i}", "correct_scale": f, "correct_strict": f}
        for i, f in enumerate(flags)
    }


def test_compare_detects_a_paired_improvement():
    a = rows([True] * 80 + [False] * 20)
    b = rows([True] * 60 + [False] * 40)
    result = compare(a, b)
    assert (result["accuracy_a"], result["accuracy_b"]) == (0.8, 0.6)
    assert result["paired_bootstrap"]["diff"] == pytest.approx(0.2)
    assert result["mcnemar"]["only_a"] == 20 and result["mcnemar"]["only_b"] == 0
    assert result["mcnemar"]["p_value"] < 0.001


def test_compare_pairs_by_id_not_by_file_order():
    a = rows([True, False, True])
    b = dict(reversed(list(rows([True, False, True]).items())))
    assert compare(a, b)["mcnemar"] == {"only_a": 0, "only_b": 0, "p_value": 1.0}


def test_compare_refuses_runs_over_different_questions():
    with pytest.raises(ValueError, match="not in both runs"):
        compare(rows([True, True]), rows([True]))


def test_compare_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        compare(rows([True]), rows([True]), metric="ungated")


def test_main_writes_result(tmp_path):
    for name, flags in (("a", [True, True, False]), ("b", [True, False, False])):
        with (tmp_path / f"{name}.jsonl").open("w") as fh:
            for row in rows(flags).values():
                fh.write(json.dumps(row) + "\n")
    out = tmp_path / "cmp.json"
    main(["--a", str(tmp_path / "a.jsonl"), "--b", str(tmp_path / "b.jsonl"), "--out", str(out)])
    saved = json.loads(out.read_text())
    assert saved["n"] == 3 and saved["mcnemar"]["only_a"] == 1
