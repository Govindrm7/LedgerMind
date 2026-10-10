"""Build the dashboard data file from the committed results.

Every number on the dashboard is computed here from files in ``docs/results`` (and the
prepared FinQA data for the explorer documents), so the page never carries a number that
cannot be traced to an artifact.

Usage::

    python -m ledgermind.data.prepare            # once, for the explorer documents
    python scripts/build_dashboard.py            # writes docs/dashboard/data.json
"""

from __future__ import annotations

import gzip
import json
import random
import subprocess
from pathlib import Path

from ledgermind.data.prepare import read_jsonl
from ledgermind.eval.compare import compare, load_rows
from ledgermind.eval.cost import api_cost_per_1k, self_hosted_cost_per_1k
from ledgermind.eval.stats import bootstrap_ci
from ledgermind.verifier.verifier import verify

RESULTS = Path("docs/results")
EVAL = RESULTS / "eval"
FRONTIER = RESULTS / "frontier"
OPEN = RESULTS / "open"
DATA = Path("data/processed/finqa")
OUT = Path("docs/dashboard/data.json")

# Self hosted cost basis: one on demand H200 at a single GPU neocloud rate (Jarvislabs,
# $3.80 per GPU hour, April 2026 survey). The page shows the $2.00 to $10.60 market range.
H200_USD_PER_HOUR = 3.80
H200_PRICE_RANGE = (2.00, 10.60)
H200_PRICE_SOURCE = "https://jarvislabs.ai/blog/h200-price"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_completions(path: Path) -> dict[str, dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def interval(flags: list[bool]) -> dict:
    ci = bootstrap_ci([float(f) for f in flags])
    return {"mean": round(ci.mean, 4), "low": round(ci.low, 4), "high": round(ci.high, 4)}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def bench_levels(name: str) -> list[dict]:
    data = read_json(RESULTS / "bench" / f"{name}.json")
    rows = data.get("levels") or data.get("results")
    keep = (
        "concurrency",
        "requests_per_s",
        "output_tokens_per_s",
        "latency_p50_s",
        "latency_p95_s",
    )
    return [{k: r[k] for k in keep} for r in rows]


def api_cost(completions: Path, usd_in: float, usd_out: float) -> dict:
    rows = list(read_completions(completions).values())
    prompt = sum(r["prompt_tokens"] or 0 for r in rows) / len(rows)
    completion = sum(r["completion_tokens"] or 0 for r in rows) / len(rows)
    return {
        "usd_per_1k": round(api_cost_per_1k(prompt, completion, usd_in, usd_out), 3),
        "basis": f"list price ${usd_in}/${usd_out} per 1M tokens, "
        f"mean {prompt:.0f} in / {completion:.0f} out tokens per query",
    }


def systems_on_sample(sample: set[str]) -> list[dict]:
    """Every system scored on the shared 400 question sample, paired against GRPO."""
    fp8 = next(r for r in bench_levels("grpo_fp8") if r["concurrency"] == 64)
    ours_cost = {
        "usd_per_1k": round(self_hosted_cost_per_1k(H200_USD_PER_HOUR, fp8["requests_per_s"]), 4),
        "basis": f"one H200 at ${H200_USD_PER_HOUR:.2f}/h, fp8, measured "
        f"{fp8['requests_per_s']} req/s at concurrency 64",
    }
    specs = [
        (
            "grpo",
            "LedgerMind GRPO (ours, 4B)",
            "ours",
            EVAL / "grpo_test_rows.jsonl",
            True,
            ours_cost,
        ),
        ("sft", "LedgerMind SFT (ours, 4B)", "ours", EVAL / "sft_bi1_test_rows.jsonl", True, None),
        (
            "base",
            "Qwen3-4B base, zero shot",
            "baseline",
            EVAL / "base_zeroshot_bi_test_rows.jsonl",
            True,
            None,
        ),
        (
            "gpt55_direct",
            "gpt-5.5, direct answer",
            "api",
            FRONTIER / "gpt55_sample400_direct_stated_rows.jsonl",
            False,
            api_cost(FRONTIER / "gpt-5.5-2026-04-23_test_sample400_direct.jsonl.gz", 5.00, 30.00),
        ),
        (
            "gpt55_pipeline",
            "gpt-5.5 + LedgerMind pipeline",
            "api",
            FRONTIER / "gpt55_sample400_pipeline_rows.jsonl",
            True,
            api_cost(FRONTIER / "gpt-5.5-2026-04-23_test_sample400_pipeline.jsonl.gz", 5.00, 30.00),
        ),
        (
            "mini_direct",
            "gpt-5.4-mini, direct answer",
            "api",
            FRONTIER / "gpt54mini_test_direct_stated_rows.jsonl",
            False,
            api_cost(FRONTIER / "gpt-5.4-mini-2026-03-17_test_direct.jsonl.gz", 0.75, 4.50),
        ),
        (
            "mini_pipeline",
            "gpt-5.4-mini + LedgerMind pipeline",
            "api",
            FRONTIER / "gpt54mini_test_pipeline_rows.jsonl",
            True,
            api_cost(FRONTIER / "gpt-5.4-mini-2026-03-17_test_pipeline.jsonl.gz", 0.75, 4.50),
        ),
    ]
    for m in open_systems().get("sample_systems", []):
        specs.append((m["key"], m["label"], "open", OPEN / m["rows"], m["verified"], None))
    grpo = {k: v for k, v in load_rows(specs[0][3]).items() if k in sample}
    out = []
    for key, label, group, rows_path, verified, cost in specs:
        rows = {k: v for k, v in load_rows(rows_path).items() if k in sample}
        assert set(rows) == sample, f"{key} does not cover the sample"
        entry = {
            "key": key,
            "label": label,
            "group": group,
            "verified_answers": verified,
            "accuracy": interval([rows[i]["correct_scale"] for i in sorted(rows)]),
            "cost": cost,
        }
        if key != "grpo":
            c = compare(grpo, rows)
            entry["grpo_minus_this"] = {**c["paired_bootstrap"], "mcnemar": c["mcnemar"]}
        out.append(entry)
    return out


def open_systems() -> dict:
    """Open weight baselines, listed in docs/results/open/systems.json when they exist."""
    path = OPEN / "systems.json"
    return read_json(path) if path.exists() else {}


def full_test_table() -> list[dict]:
    specs = [
        ("LedgerMind GRPO (ours)", EVAL / "grpo_test.json"),
        ("LedgerMind GRPO, fp8 (ours)", EVAL / "grpo_fp8_test.json"),
        ("LedgerMind SFT (ours)", EVAL / "sft_bi1_test.json"),
        ("Qwen3-4B base, zero shot", EVAL / "base_zeroshot_bi_test.json"),
        ("gpt-5.4-mini + pipeline", FRONTIER / "gpt54mini_test_pipeline.json"),
        ("gpt-5.4-mini, direct (stated precision)", FRONTIER / "gpt54mini_test_direct_stated.json"),
        ("gpt-5.4-mini, direct (default scoring)", FRONTIER / "gpt54mini_test_direct_default.json"),
    ]
    specs += [(m["label"], OPEN / m["summary"]) for m in open_systems().get("full_test", [])]
    rows = []
    for label, path in specs:
        d = read_json(path)
        rows.append(
            {
                "label": label,
                "n": d["n"],
                "verified": d["verified_accuracy"],
                "strict": d["verified_accuracy_strict"]["mean"],
                # ungated accuracy (every computed value returned) only applies to pipeline answers
                "ungated": None if d.get("direct_matching") else d["ungated_accuracy"]["mean"],
                "coverage": d["coverage"]["mean"],
                "parse_rate": d["parse_rate"],
                "fabrication_rate": d["fabrication_rate"],
                "latency_p50_s": d["latency_s"]["p50"],
                "completion_tokens_mean": round(d["completion_tokens_mean"], 1),
            }
        )
    return rows


def sharpening() -> dict:
    greedy = {
        m: read_json(EVAL / f"{f}.json")["verified_accuracy"]["mean"]
        for m, f in (("sft", "sft_bi1_test"), ("grpo", "grpo_test"))
    }
    runs = {
        m: [load_rows(EVAL / f"{m}_test_t0.9_s{s}_rows.jsonl") for s in range(4)]
        for m in ("sft", "grpo")
    }
    sampled = {
        m: sum(
            read_json(EVAL / f"{m}_test_t0.9_s{s}.json")["verified_accuracy"]["mean"]
            for s in range(4)
        )
        / 4
        for m in runs
    }
    ids = sorted(runs["sft"][0])
    consistency = {}
    for m, seeds in runs.items():
        hits = [sum(bool(s[i]["correct_scale"]) for s in seeds) for i in ids]
        consistency[m] = {
            "all_4": sum(h == 4 for h in hits),
            "some": sum(0 < h < 4 for h in hits),
            "none": sum(h == 0 for h in hits),
        }
    pooled = read_json(RESULTS / "compare" / "compare_grpo_vs_sft_sampled_t0.9.json")[
        "pooled_grpo_minus_sft"
    ]
    greedy_cmp = read_json(RESULTS / "compare" / "compare_grpo_vs_sft_correct_scale.json")[
        "paired_bootstrap"
    ]
    return {
        "greedy": greedy,
        "sampled": sampled,
        "consistency": consistency,
        "n": len(ids),
        "greedy_mcnemar_p": read_json(
            RESULTS / "compare" / "compare_grpo_vs_sft_correct_scale.json"
        )["mcnemar"]["p_value"],
        "sampled_diff": pooled,
        "greedy_diff": greedy_cmp,
    }


def training() -> dict:
    grpo = read_json(RESULTS / "training" / "grpo_run.json")["log_history"]
    sft = read_json(RESULTS / "training" / "sft_run.json")["log_history"]
    grpo_points = [
        {
            k: round(float(h[s]), 4)
            for k, s in (
                ("step", "step"),
                ("correct", "reward/correct"),
                ("entropy", "entropy"),
                ("zero_std", "frac_reward_zero_std"),
                ("length", "completions/mean_length"),
            )
        }
        for h in grpo
        if "reward/correct" in h
    ]
    sft_eval = [
        {"step": h["step"], "eval_loss": round(float(h["eval_loss"]), 5)}
        for h in sft
        if "eval_loss" in h
    ]
    return {"grpo": grpo_points, "sft_eval": sft_eval}


def saboteur() -> dict:
    s = read_json(RESULTS / "saboteur_test.json")
    mode_a = {
        k: {"cases": v["cases"], "rate": v["detection_rate"]}
        for k, v in s["mode_a_output_faults"].items()
    }
    mode_b = {
        k: {
            "stale_rejection_rate": v["stale_rejection_rate"],
            "faithful_detection_rate": v["faithful_detection_rate"],
            "detection_rate_when_detectable": v.get("detection_rate_when_detectable"),
        }
        for k, v in s["mode_b_document_faults"].items()
    }
    mode_c = []
    pairs = [
        (
            "Qwen3-4B base",
            EVAL / "base_zeroshot_bi_test",
            EVAL / "base_zeroshot_counterfactual_test",
        ),
        ("LedgerMind SFT", EVAL / "sft_bi1_test", EVAL / "sft_counterfactual_test"),
        ("LedgerMind GRPO", EVAL / "grpo_test", EVAL / "grpo_counterfactual_test"),
    ]
    pairs += [
        (m["label"], OPEN / m["real"], OPEN / m["counterfactual"])
        for m in open_systems().get("mode_c", [])
    ]
    for label, real, cf in pairs:
        cf_rows = load_rows(Path(f"{cf}_rows.jsonl"))
        real_rows = {k: v for k, v in load_rows(Path(f"{real}_rows.jsonl")).items() if k in cf_rows}
        c = compare(cf_rows, real_rows)
        mode_c.append(
            {
                "label": label,
                "n": len(cf_rows),
                "real": c["accuracy_b"],
                "counterfactual": c["accuracy_a"],
                "diff": c["paired_bootstrap"],
                "mcnemar_p": c["mcnemar"]["p_value"],
                "recalled_original": sum(r["recalled_original"] for r in cf_rows.values()),
            }
        )
    return {"mode_a": mode_a, "mode_b": mode_b, "mode_c": mode_c}


def doc_payload(ex) -> dict:
    return {"table": [list(r) for r in ex.document.table], "sentences": list(ex.document.sentences)}


def pipeline_view(completion: str, doc) -> dict:
    return verify(completion, doc).audit()


def explorer(sample: set[str], per_group: int = 6) -> list[dict]:
    """A seeded selection of questions per outcome, with full audit trails."""
    test = {e.id: e for e in read_jsonl(DATA / "test.jsonl")}
    cf = {e.id: e for e in read_jsonl(DATA / "counterfactual_test.jsonl")}
    grpo_rows = load_rows(EVAL / "grpo_test_rows.jsonl")
    grpo_out = read_completions(EVAL / "grpo_test.jsonl.gz")
    g55p_rows = load_rows(FRONTIER / "gpt55_sample400_pipeline_rows.jsonl")
    g55d_rows = load_rows(FRONTIER / "gpt55_sample400_direct_stated_rows.jsonl")
    g55p_out = read_completions(FRONTIER / "gpt-5.5-2026-04-23_test_sample400_pipeline.jsonl.gz")
    g55d_out = read_completions(FRONTIER / "gpt-5.5-2026-04-23_test_sample400_direct.jsonl.gz")
    cf_rows = load_rows(EVAL / "grpo_counterfactual_test_rows.jsonl")
    cf_out = read_completions(EVAL / "grpo_counterfactual_test.jsonl.gz")
    ids = sorted(sample)

    def ok(rows, i):
        return bool(rows[i]["correct_scale"])

    groups = {
        "both_correct": [i for i in ids if ok(grpo_rows, i) and ok(g55d_rows, i)],
        "ours_only": [i for i in ids if ok(grpo_rows, i) and not ok(g55d_rows, i)],
        "gpt55_only": [i for i in ids if not ok(grpo_rows, i) and ok(g55d_rows, i)],
        "rejected": [i for i in sorted(grpo_rows) if not grpo_rows[i]["accepted"]],
    }
    titles = {
        "both_correct": "Both correct",
        "ours_only": "LedgerMind right, gpt-5.5 direct wrong",
        "gpt55_only": "gpt-5.5 direct right, LedgerMind wrong",
        "rejected": "Rejected by the verifier",
        "counterfactual": "Counterfactual document (edited figures)",
    }
    rng = random.Random(0)
    picked = {g: sorted(rng.sample(v, min(per_group, len(v)))) for g, v in groups.items()}
    picked["counterfactual"] = sorted(
        rng.sample([i for i in sorted(cf_rows) if cf_rows[i]["correct_scale"]], per_group)
    )
    items = []
    for group, chosen in picked.items():
        for i in chosen:
            is_cf = group == "counterfactual"
            ex = cf[i] if is_cf else test[i]
            out = cf_out[i] if is_cf else grpo_out[i]
            item = {
                "id": i,
                "group": group,
                "group_title": titles[group],
                "question": ex.question,
                "gold": ex.gold_answer,
                "document": doc_payload(ex),
                "ours": {
                    **pipeline_view(out["completion"], ex.document),
                    "correct": ok(cf_rows if is_cf else grpo_rows, i),
                },
            }
            if is_cf:
                item["original_gold"] = test[i].gold_answer
            if not is_cf and i in g55p_out:
                item["gpt55_pipeline"] = {
                    **pipeline_view(g55p_out[i]["completion"], ex.document),
                    "correct": ok(g55p_rows, i),
                }
                text = g55d_out[i]["completion"].strip()
                item["gpt55_direct"] = {
                    "text": text[-600:],
                    "value": g55d_rows[i]["value"],
                    "correct": ok(g55d_rows, i),
                }
            items.append(item)
    return items


def main() -> None:
    sample = set(read_json(FRONTIER / "test_sample400_ids.json")["ids"])
    spend = read_json(FRONTIER / "spend.json")
    data = {
        "meta": {
            "git": git_sha(),
            "sample_n": len(sample),
            "test_n": 1139,
            "api_spend_usd": round(spend["spent_usd"], 2),
            "h200_usd_per_hour": H200_USD_PER_HOUR,
            "h200_price_range": H200_PRICE_RANGE,
            "h200_price_source": H200_PRICE_SOURCE,
        },
        "sample_systems": systems_on_sample(sample),
        "full_test": full_test_table(),
        "sharpening": sharpening(),
        "training": training(),
        "serving": {
            name: bench_levels(name)
            for name in ("grpo_bf16", "grpo_fp8", "grpo_bf16_constrained", "grpo_fp8_constrained")
        },
        "saboteur": saboteur(),
        "explorer": explorer(sample),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(
        f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, "
        f"{len(data['explorer'])} explorer questions)"
    )


if __name__ == "__main__":
    main()
