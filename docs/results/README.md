# Results index

Every number in the README and DESIGN.md comes from a file in this folder. This page says which run produced each file and with what settings.

All runs used one NVIDIA H200 on the Northeastern Explorer cluster, with torch 2.13.0+cu129, vLLM 0.31.0, transformers 5.17.0, TRL 1.15.0 and PEFT 0.21.2 (pinned in `slurm/requirements-cu129.lock`). Base model: `Qwen/Qwen3-4B-Base`.

## Conventions

* **Test set:** the 1,139 FinQA test questions whose gold program grounds fully (`data/processed/finqa/test.jsonl`). **Counterfactual set:** the 846 test questions that Saboteur Mode C could edit (`counterfactual_test.jsonl`).
* **Verified accuracy:** the answer is correct *and* the verifier accepted it, over all questions. Scale tolerant unless marked strict. Intervals are 95% bootstrap intervals.
* **Batch invariant:** scored runs set `VLLM_BATCH_INVARIANT=1`, so greedy decoding gives identical outputs on every run. Two SFT runs (`sft_bi1`, `sft_bi2`) produced 0 differing completions out of 1,139. Paired tests rely on this.
* **Logged vs code commit:** "logged" is the cluster checkout's HEAD that the job printed. Some early jobs ran with files copied to the cluster before they were committed, so "code" names the commit whose files match what actually ran.

## Evaluation runs (`eval/`)

Each run has a summary (`<name>.json`), per question verdicts (`<name>_rows.jsonl`) and the raw model outputs (`<name>.jsonl.gz`).

| File prefix | System | Set | Decoding | Job | Logged | Code |
|---|---|---|---|---|---|---|
| `base_zeroshot_bi_test` | Qwen3-4B-Base, zero shot | test | greedy, bf16 | 10935260 | 57c1543 | d7e2461 |
| `sft_bi1_test` | Base + SFT | test | greedy, bf16 | 10935175 | 57c1543 | d9539e5 |
| `sft_bi2_test` (outputs only) | Base + SFT, repeat of `sft_bi1` | test | greedy, bf16 | 10935176 | 57c1543 | d9539e5 |
| `grpo_test` | Base + SFT + GRPO | test | greedy, bf16 | 10940414 | aafa72d | aafa72d |
| `grpo_fp8_test` | Base + SFT + GRPO | test | greedy, fp8 | 10948441 | 3ae6555 | 3ae6555 |
| `sft_test_t0.9_s{0..3}` | Base + SFT | test | sampled, temperature 0.9, seeds 0 to 3 | 10945538 | aafa72d | 3ae6555 |
| `grpo_test_t0.9_s{0..3}` | Base + SFT + GRPO | test | sampled, temperature 0.9, seeds 0 to 3 | 10945539 | aafa72d | 3ae6555 |
| `base_zeroshot_counterfactual_test` | Qwen3-4B-Base, zero shot | counterfactual | greedy, bf16 | 10948436 | 3ae6555 | 3ae6555 |
| `sft_counterfactual_test` | Base + SFT | counterfactual | greedy, bf16 | 10948437 | 3ae6555 | 3ae6555 |
| `grpo_counterfactual_test` | Base + SFT + GRPO | counterfactual | greedy, bf16 | 10948438 | 3ae6555 | 3ae6555 |

The counterfactual set was rebuilt on the cluster (job 10948435, code 2fc7cce) and is byte identical to the local build (SHA256 `d9b2bd94...`).

## Paired comparisons (`compare/`)

Produced by `python -m ledgermind.eval.compare` from the rows files above (McNemar and paired bootstrap on the same questions).

| File | A vs B |
|---|---|
| `compare_sft_vs_base_correct_{scale,strict}.json` | `sft_bi1_test` vs `base_zeroshot_bi_test` |
| `compare_grpo_vs_sft_correct_{scale,strict}.json` | `grpo_test` vs `sft_bi1_test` |
| `compare_grpo_fp8_vs_bf16_correct_{scale,strict}.json` | `grpo_fp8_test` vs `grpo_test` |
| `compare_grpo_vs_sft_sampled_t0.9.json` | sampled GRPO vs sampled SFT, per question mean over seeds 0 to 3 |

## Serving benchmark (`bench/`)

GRPO model served by vLLM in its default mode (not batch invariant), one H200, 256 requests per concurrency level (1, 4, 16, 64) after a warmup, unconstrained and with JSON schema decoding.

| File | Variant | Job | Logged | Code |
|---|---|---|---|---|
| `grpo_bf16.json`, `grpo_bf16_constrained.json` | bf16 | 10948439 | 3ae6555 | 9bfc178 |
| `grpo_fp8.json`, `grpo_fp8_constrained.json` | fp8 | 10948440 | 3ae6555 | 9bfc178 |

## Training (`training/`)

| File | What | Job | Code |
|---|---|---|---|
| `sft_run.json` | SFT config, loss and eval history | 10933900 | 57c1543 |
| `pass_rates.json` | SFT pass rate per training prompt (k=8, temperature 0.9) used to pick GRPO prompts | 10935259 | 3ddc39c |
| `grpo_run.json` | GRPO config, per step rewards by term, entropy and lengths | 10939642 | aafa72d |

## Reproducing a number

```bash
# Paired test from stored verdicts (no GPU needed)
python -m ledgermind.eval.compare --a docs/results/eval/grpo_test_rows.jsonl --a-name grpo \
    --b docs/results/eval/sft_bi1_test_rows.jsonl --b-name sft

# Rescore stored outputs with the current verifier
gunzip -k docs/results/eval/grpo_test.jsonl.gz
python -m ledgermind.eval.harness --examples data/processed/finqa/test.jsonl \
    --predictions docs/results/eval/grpo_test.jsonl --name grpo --out /tmp/grpo_rescored.json
```
