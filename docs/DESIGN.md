# LedgerMind design

This document explains how LedgerMind works, why each piece exists, and what has been measured so far. Every number here comes from a script in this repository and a result file in [`docs/results/`](results/).

## 1. Problem

Language models are fluent at financial text and unreliable at financial arithmetic. Errors come in three kinds:

1. **Fabrication.** The model states a figure that is not in the filing.
2. **Misreading.** The figure exists but comes from the wrong row, the wrong period, the wrong sign or the wrong scale.
3. **Arithmetic.** The figures are right but the computation is wrong.

A human reviewer can't tell these apart from the final number alone. LedgerMind restructures the task so each kind of error becomes mechanically checkable.

## 2. Approach

The model never produces the final number. It produces a claim that can be audited:

```
Question + Document
      │
      ▼
[1] Model (SFT, then GRPO)  ──►  evidence: each figure with a pointer to a table cell or sentence
                                  plan:     an expression over the evidence, e.g. (e1 - e2) / e2
      │
      ▼
[2] Executor  ──►  computes the number from the plan (safe AST evaluation, whitelisted operations)
      │
      ▼
[3] Verifier  ──►  provenance, plan hygiene, execution, invariants, document integrity
      │
      ▼
[4] Answer with an audit trail, or a rejection with reasons
```

Fabrication and misreading are caught by provenance. Arithmetic errors become impossible, because a deterministic executor does the arithmetic. What is left for the model to get right is choosing the right figures and the right operations, and that is exactly the part reinforcement learning optimizes.

The same `verify()` function serves as the inference gate, the GRPO reward signal and the eval scorer, so training, serving and evaluation can never disagree about what "correct" means.

## 3. Output contract

```json
{
  "reasoning": "Step 1: subtract e2 from e1. Step 2: divide result 1 by e2.",
  "evidence": [
    {"id": "e1", "value": 5829, "label": "2016 net revenue (amount)", "source": {"type": "table", "row": 6, "col": 1}},
    {"id": "e2", "value": 5735, "label": "2015 net revenue (amount)", "source": {"type": "table", "row": 1, "col": 1}}
  ],
  "plan": "(e1 - e2) / e2",
  "answer_unit": "ratio"
}
```

Or an explicit refusal: `{"abstain": true, "reason": "..."}`.

Design decisions:

* **There is no field for the answer.** Unknown fields are rejected (`extra="forbid"`), so `"answer": 0.0164` fails validation instead of being silently trusted.
* **Values are copied exactly as printed.** For `12%` the model writes 12; for `$ -9,457` it writes -9457. Conversions are explicit in the plan (`e1 / 100`, `e1 * 1000000`, `(-e1)`). Nothing is converted implicitly, because implicit conversion is where scale and sign errors hide.
* **The plan language is tiny.** It allows `+ - * /`, the functions `add sub mul div pow pct_change sum mean max min greater`, evidence ids, and a short list of unit and scale constants (1 to 10, 12, 100, 365, 1000, 10^4, 10^5, 10^6, 10^9). Any other literal is rejected, so a plan can't carry a precomputed answer.

## 4. The verifier

| Gate | What it checks | Rejects as |
|---|---|---|
| Schema | Output parses as an Answer or an Abstain | `schema_invalid` |
| Provenance | Each cited value is printed at its cited location, sign included | `fabricated`, `misattributed`, `bad_pointer` |
| Plan hygiene | Plan compiles; references only defined evidence; **the result moves when evidence moves**; unit type matches | `bypass`, `unknown_ref`, `plan_invalid`, `unit_mismatch` |
| Execution | Executor computes a finite value | `execution_error` |
| Invariants | Units, scales and periods are combined consistently | `unit_mismatch`, `scale_mismatch`, `period_mismatch` |
| Integrity | Cited cells are not part of a broken table footing | `source_footing_broken` |

**The sensitivity check** in plan hygiene closes a loophole the literal allowlist leaves open. `e1 - e1 + 12` uses only allowed constants and references evidence, but its result never depends on the evidence. The verifier perturbs each referenced value in turn, and if the result never moves, the plan is a disguised constant.

**Invariants** work as dimensional analysis. Each figure gets a kind (percent if printed with %, currency if printed with $), a scale (from a scale word next to it, the row label, the column header or the table's corner cell) and a period (quarter or annual, from the column header). Additive operations on incompatible dimensions are violations. Unknown dimensions are compatible with anything, so the check only fires on positive evidence of a mismatch.

**Integrity** checks the document, not the answer. Real tables foot: line items sum to their total in every amount column with the same block of rows. If other columns foot and one does not, that column was altered. The check handles nested subtotals, nil cells and non-additive columns (averages, prices, rates).

### False rejection cost

Applied to the converted gold targets, which are correct by construction, the full verifier rejects:

| Split | Gold targets | Rejected | Main causes |
|---|---|---|---|
| dev | 873 | 0.57% | footing 2, scale mismatch 2, unit mismatch 1 |
| test | 1,139 | 0.79% | footing 8, unit mismatch 1 |

Several of the flagged gold programs are themselves wrong upstream (for example, they add a value in billions to one in millions). The integrity check also fires on about 1% of clean documents, which is why it only rejects answers that cite a flagged cell.

## 5. The Saboteur: adversarial testing

The Saboteur attacks the system in three ways. Results are on FinQA test, seed 0 ([`saboteur_test.json`](results/saboteur_test.json)).

### Mode A: corrupted outputs

Correct answers are mutated with the failure modes language models actually show:

| Fault | Cases | Caught |
|---|---|---|
| One digit of a cited value changed | 1,137 | 100% |
| Sign of a cited value flipped | 1,139 | 100% |
| Pointer moved to another row | 832 | 100% |
| Pointer moved to another column (period mixup) | 677 | 100% |
| Value scaled by 1000 | 1,139 | 100% |
| Percent written as a fraction | 63 | 100% |
| Plan replaced by the final number | 1,106 | 100% |
| Plan replaced by `e1 - e1 + k` | 1,119 | 100% |

Mode A is a guarantee, not a learned detector: provenance is an exact check, so these catch rates follow by construction. The test's job is to prove that no code path lets a corrupted claim through.

### Mode B: corrupted documents

The source document is altered at a cited cell. Two different questions are asked:

* **Stale answer.** If an answer computed from the original document is checked against the altered one, is it rejected? Yes: 100% for digit changes, sign flips and column swaps, and 99.65% for row deletions (the remainder are cases where a duplicate row moves into the cited position with the same printed value, so the answer is still literally true of the document).
* **Faithful reader.** If a model correctly reads the *altered* document, provenance passes, and it should. Only the document's internal redundancy can reveal the tampering:

| Fault | Flagged overall | Flagged when the cell is in a footed block | Share of cases with a footed block |
|---|---|---|---|
| Digit change | 25.2% | 77.0% | 31.8% |
| Sign flip | 27.0% | 84.6% | 31.5% |
| Column swap | 23.3% | 65.0% | 34.7% |
| Row deletion | 0.7% (document level) | n/a | n/a |

This is reported as is. A changed figure with no total and no restatement is undetectable from the document alone, as is a change smaller than printed rounding (a total may legitimately differ from the sum of rounded items by half a unit per item). Deleting a whole row breaks every column equally, which the cross-column rule can't distinguish from a total that was never meant to foot.

### Mode C: counterfactual documents

FinQA is built from public 10-K filings that frontier models have likely seen during pretraining. Mode C rescales each figure a question depends on by 3 to 10%, rewrites every printed copy of that figure in its original format, and recomputes the gold answer. Edits that would break a footing are discarded, so the 846 counterfactual test documents pass every integrity check a real filing passes.

A system that reads the document gets the new answer. A system that recalls the original filing gets the old one, and the eval harness reports this as `recalled_original_rate`. LedgerMind's provenance gate rejects recalled figures, because they are no longer printed where they're cited.

## 6. Data

FinQA (Chen et al., EMNLP 2021) is loaded from the upstream repository at a pinned commit, with SHA256 verification. Gold programs such as `subtract(5829, 5735), divide(#0, 5735)` are converted into the output contract by locating every argument in the document: supporting facts first, then the whole document, preferring distinct cells when one step uses the same value twice.

| Split | Examples | Grounded | Oracle accuracy on grounded | Pipeline ceiling |
|---|---|---|---|---|
| train | 6,251 | 99.28% | 99.98% | 99.26% |
| dev | 883 | 98.87% | 100.00% | 98.87% |
| test | 1,147 | 99.30% | 100.00% | 99.30% |

"Oracle" runs the converted gold plan through the executor and compares the result with the gold answer. 100% on dev and test validates the number parser, the converter, the executor and the answer matcher together. The single train mismatch is an upstream label error.

Dataset quirks found and handled, each with a test:

* `9%` in a gold program means 0.09, and table operations over percent cells divide by 100.
* Gold answers are rounded to five decimals. Answers within `max(1e-5, 1e-4 × |gold|)` count as matches.
* Percentages are often stored as fractions (0.0162 for 1.62%). Every system is scored both strictly and with a scale-tolerant match.
* The processed text repeats values in echo groups (`$ -23158 ( 23158 )`), and an em dash in a table cell appears as the digits `2014` (Unicode U+2014). Apostrophes show up as `2019` the same way.
* The "original" table differs in shape from the processed one in about 17% of examples, so the processed table is canonical.
* Twelve gold programs are degenerate (`subtract(x, x)` over a single printed value) and are excluded so that SFT never learns a bypass pattern.

## 7. Training

**Stage 1, SFT.** Teaches the contract. A LoRA adapter (rank 32, alpha 64) is trained on prompt and completion pairs, with loss on the JSON target only; the trainer asserts that completions end with EOS. Prompts are about 1,650 tokens at the median and 2,330 at p95, and targets about 110 tokens, which sets `max_length` 4096.

**Stage 2, GRPO with verifiable rewards.** The SFT adapter is merged into the base weights and a fresh adapter is trained. The reward is the verifier, so there is no reward model:

| Term | Weight | When |
|---|---|---|
| correct | +1.0 | answer matches gold, with no fabricated figure and no bypass |
| provenance | +0.3 | every evidence item the plan uses passes provenance |
| schema | +0.1 | output parses |
| fabrication | -0.5 | any cited figure is fabricated, misattributed or dangling |
| bypass | -1.0 | plan does not compute from evidence, or a final answer field is smuggled in |
| unused evidence | -0.05 each | cited but unused (capped at -0.25) |
| invariant | -0.2 | unit, scale or period mismatch |

The resulting order of outcomes is: correct and grounded (1.4) > grounded but wrong (0.4) > abstain (0.1) > unparseable (0) > fabricated > bypass (-0.9). Tests pin this order. On dev, 870 of 873 gold targets score the maximum; the other three are the noisy upstream labels the invariant checks flag.

Decisions and their reasons:

* **Fabricated figures can't earn correctness.** Citing the right numbers at the wrong place yields the right value but no credit.
* **The provenance bonus only counts evidence the plan uses**, and unused evidence costs reward, so padding an answer with easy, real figures does not pay.
* **Learning rate 5e-6 for LoRA**, about 10x a full finetune rate, with a sweep from 2e-6 to 1e-5 planned.
* **Prompt filtering by SFT pass rate.** A prompt where all 8 rollouts score the same contributes zero advantage. The CPU smoke test shows this vividly: a random model yields 100% invalid outputs, `frac_reward_zero_std = 1` and a loss of exactly zero. Only prompts the SFT model solves sometimes, but not always, are kept.
* **beta 0 and the DAPO loss** (TRL 1.x defaults), with truncated completions masked. A KL penalty of 0.04 is an ablation.
* **Per-term logging.** Every reward term and verdict share is logged separately, so a reward hack shows up as one term moving while `reward/correct` does not.

Both stages run end to end in CPU smoke tests with a tiny model. TRL 1.x routes the loss through a Triton kernel that ships only on Linux; the smoke tests install a pure PyTorch fallback that GPU runs never use.

## 8. Evaluation

| Metric | Definition |
|---|---|
| Verified accuracy | Accepted by the verifier and correct, over all questions. The headline. |
| Coverage | Share of questions the verifier lets through |
| Selective accuracy | Correct among accepted |
| Ungated accuracy | Correct if every executed value were returned. The gap to verified accuracy is the cost of the gate. |
| Fabrication rate | Parsed answers citing a figure not printed where cited |
| Recalled original rate | On counterfactual sets, answers matching the real filing instead of the edited document |
| Latency, throughput | p50 and p95 per request; requests and tokens per second under a concurrency sweep |
| Cost | $ per 1,000 queries: GPU hourly price over measured throughput, against API list prices on measured token counts |

Every metric carries a bootstrap 95% interval (about ±2.5 points at n = 1,139), and every comparison uses a paired test on the same questions (paired bootstrap and exact McNemar). Gold targets score 99.21% verified accuracy on test [98.68%, 99.65%], which matches the measured false rejection rate.

Frontier models are compared two ways: answering directly with chain of thought, and running through the same LedgerMind pipeline. The second comparison is the fair one. If it beats the fine-tuned model on accuracy, the result is reported that way.

## 9. Reproducing

```bash
uv sync                                                  # core package, CPU only
uv run python -m ledgermind.data.prepare                 # download, verify, convert FinQA
uv run python -m ledgermind.eval.oracle                  # oracle check
uv run python -m ledgermind.saboteur.run --split test --out docs/results/saboteur_test.json
uv run python -m ledgermind.saboteur.counterfactual --split test
uv run python -m ledgermind.eval.harness --examples data/processed/finqa/test.jsonl \
    --oracle --name oracle --out docs/results/eval_oracle_test.json

uv sync --extra train                                    # training (CUDA for real runs)
uv run python training/sft.py --config training/configs/sft_qwen3_4b.yaml
uv run python training/merge_adapter.py --base Qwen/Qwen3-4B-Base \
    --adapter checkpoints/sft_qwen3_4b --out checkpoints/sft_qwen3_4b_merged
python training/pass_rate.py --model checkpoints/sft_qwen3_4b_merged \
    --train-file data/processed/finqa/train.jsonl --out outputs/pass_rates.json
uv run python training/grpo.py --config training/configs/grpo_qwen3_4b.yaml

serving/launch_vllm.sh checkpoints/grpo_qwen3_4b_merged bf16
uv run python -m ledgermind.serving.predict --examples data/processed/finqa/test.jsonl \
    --base-url http://localhost:8000/v1 --model ledgermind --out outputs/pred_grpo.jsonl
uv run python -m ledgermind.serving.bench --base-url http://localhost:8000/v1 \
    --model ledgermind --examples data/processed/finqa/test.jsonl --out docs/results/bench.json
```

## 10. Limitations

* FinQA covers single page excerpts with a gold document. Retrieval over full filings is out of scope.
* Integrity checks need internal redundancy (totals, restatements). About two thirds of tampered cells sit outside any footed block and are undetectable from the document alone.
* Provenance proves that a figure is printed where it is cited, not that it is the semantically right figure for the question. Choosing the right figure is what the model is trained for, and verified accuracy measures it.
* Gold labels carry upstream noise. Known cases are excluded or reported, never silently fixed.
