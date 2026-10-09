# LedgerMind

[![CI](https://github.com/Govindrm7/LedgerMind/actions/workflows/ci.yml/badge.svg)](https://github.com/Govindrm7/LedgerMind/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**The LLM proposes. A deterministic verifier decides every number.**

LLMs are unreliable at multistep financial arithmetic, and a wrong number looks exactly like a right one. LedgerMind fine-tunes a small open model to *extract figures with source citations and write a computation plan*, never the final number. A deterministic executor runs the plan, and a verifier checks every cited figure against the source document. The system returns an answer with a full audit trail, or refuses with a reason.

The verifier is also the reward. The model is trained with SFT, then GRPO with verifiable rewards, so it is optimized directly for answers that survive verification.

```
Question + Document
      │
      ▼
 Model (SFT + GRPO)  ──►  evidence with source pointers + computation plan
      │
      ▼
 Executor            ──►  the number (safe AST evaluation, no eval)
      │
      ▼
 Verifier            ──►  provenance · plan hygiene · invariants · document integrity
      │
      ▼
 Answer + audit trail, or REJECT with reason
```

## Results so far

Model training runs on a rented GPU and is the next stage. Everything below is measured on FinQA with the verifier, data pipeline and adversarial harness already in this repo.

| What | Result |
|---|---|
| FinQA examples grounded (every figure tied to a table cell or sentence) | 99.3% of test |
| Oracle: gold programs through the executor reproduce gold answers | 100% of dev and test |
| Verifier false rejections on correct gold answers | 0.8% of test |
| Saboteur Mode A: corrupted outputs caught (7,212 injected faults, 8 fault types) | 100% |
| Saboteur Mode B: stale answers rejected after the document is tampered with | 99.6 to 100% |
| Saboteur Mode B: tampering flagged for a faithful reader, when a total covers the cell | 65 to 85% |
| Counterfactual test documents for detecting recall of real filings | 846 |

Mode A is a guarantee by construction, and the test proves no code path bypasses it. Mode B's limits are real: about two thirds of tampered cells sit outside any total and can't be detected from the document alone. [`docs/DESIGN.md`](docs/DESIGN.md) reports both.

## What the verifier checks

| Gate | Catches |
|---|---|
| **Provenance** | a figure not printed where it is cited: fabricated, wrong row, wrong period, wrong sign, wrong scale |
| **Plan hygiene** | a plan that does not really compute from the evidence, including disguised constants like `e1 - e1 + 12` (caught by a sensitivity check) |
| **Invariants** | dimensional errors: percent combined with dollars, millions with billions, quarterly with annual |
| **Integrity** | tampered source tables, via line items that no longer sum to their totals |

## The Saboteur

Adversarial testing is a first class part of the system, not an afterthought:

* **Mode A** corrupts model outputs with the errors LLMs really make, and measures what the verifier catches.
* **Mode B** corrupts the source document, and measures whether stale answers are rejected and whether tampering is visible.
* **Mode C** builds counterfactual filings: every figure a question needs is shifted by 3 to 10% and the answer recomputed. A model that recalls the real 10-K instead of reading the document gets caught.

## Repository layout

```
src/ledgermind/   numbers, schema, document, dsl (executor), verifier, saboteur,
                  data (FinQA), rewards, training helpers, eval, serving client
training/         SFT and GRPO scripts, adapter merge, pass rate filter, configs
serving/          vLLM launch script (bf16, fp8, awq)
docs/             DESIGN.md and every result file the README cites
tests/            255 tests, run in CI on Python 3.11 and 3.12
```

## Quickstart

```bash
uv sync
uv run pytest
uv run python -m ledgermind.data.prepare      # download, verify and convert FinQA
uv run python -m ledgermind.eval.oracle       # gold programs through the executor
uv run python -m ledgermind.saboteur.run --split test --out docs/results/saboteur_test.json
```

The full training, serving and benchmark commands are in [`docs/DESIGN.md`](docs/DESIGN.md#9-reproducing).

## Roadmap

* [x] Data pipeline, executor, verifier, Saboteur (Modes A, B, C), eval harness
* [x] SFT and GRPO training pipeline with verifier rewards, smoke tested end to end
* [x] vLLM serving, benchmark client and cost model
* [ ] SFT and GRPO runs on Qwen3-4B-Base
* [ ] Frontier baselines (direct and through the same pipeline)
* [ ] Quantization sweep (bf16 vs fp8) and accuracy vs cost chart

## License

Apache 2.0. See [LICENSE](LICENSE).
