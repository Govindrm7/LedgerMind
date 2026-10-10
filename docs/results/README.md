# Results index

Every number in the README and DESIGN.md comes from a file in this folder. This page says which run produced each file and with what settings.

All runs used one NVIDIA H200 on the Northeastern Explorer cluster, with torch 2.13.0+cu129, vLLM 0.31.0, transformers 5.17.0, TRL 1.15.0 and PEFT 0.21.2 (pinned in `slurm/requirements-cu129.lock`). Base model: `Qwen/Qwen3-4B-Base`.

## Conventions

* **Test set:** the 1,139 FinQA test questions whose gold program grounds fully (`data/processed/finqa/test.jsonl`). **Counterfactual set:** the 846 test questions that Saboteur Mode C could edit (`counterfactual_test.jsonl`).
* **Verified accuracy:** the answer is correct *and* the verifier accepted it, over all questions. Scale tolerant unless marked strict. Intervals are 95% bootstrap intervals.
* **Batch invariant:** scored runs set `VLLM_BATCH_INVARIANT=1`, so greedy decoding gives identical outputs on every run. Two SFT runs (`sft_bi1`, `sft_bi2`) produced 0 differing completions out of 1,139. Paired tests rely on this.
* **Scoring version:** every summary and rows file was rescored with commit 9cb55df, which fixed scale tolerant matching to compare in the gold value's units (a percent answer such as 1.71447 now matches a five decimal fraction gold such as 0.01714). Raw outputs are unchanged; only the trained models' baselines moved by more than a rounding error (base zero shot 20.02% to 20.37%).
* **Prompt versions:** the trained models and every run below used pipeline prompt v1 unless marked v2. v1 leaves the meaning of function arguments undefined (for example `pct_change(old, new)`) and tells models to convert millions to units, while FinQA answers stay in document units. Models trained on FinQA learn both conventions from data; general models follow the text. v2 (`ledgermind.data.prompts.INSTRUCTIONS_V2`) spells both out.
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
| `grpo_v2_test` | Base + SFT + GRPO, prompt v2 | test | greedy, bf16 | 10969071 | 417cf40 | 417cf40 |
| `grpo_v2_counterfactual_test` | Base + SFT + GRPO, prompt v2 | counterfactual | greedy, bf16 | 10969072 | 417cf40 | 417cf40 |
| `sft_v2_test` | Base + SFT, prompt v2 | test | greedy, bf16 | 10969073 | 417cf40 | 417cf40 |
| `sft_v2_counterfactual_test` | Base + SFT, prompt v2 | counterfactual | greedy, bf16 | 10969941 | 417cf40 | 417cf40 |
| `base_zeroshot_counterfactual_test` | Qwen3-4B-Base, zero shot | counterfactual | greedy, bf16 | 10948436 | 3ae6555 | 3ae6555 |
| `sft_counterfactual_test` | Base + SFT | counterfactual | greedy, bf16 | 10948437 | 3ae6555 | 3ae6555 |
| `grpo_counterfactual_test` | Base + SFT + GRPO | counterfactual | greedy, bf16 | 10948438 | 3ae6555 | 3ae6555 |

The counterfactual set was rebuilt on the cluster (job 10948435, code 2fc7cce) and is byte identical to the local build (SHA256 `d9b2bd94...`).

## Frontier API baselines (`frontier/`)

Run from a laptop against the OpenAI API with prompt v1, default reasoning effort, under one hard spending cap of $20 shared by all runs (`--max-cost-usd`, ledger in `spend.json`, which includes the 20 question pilot). Total spent: $16.52, no prompt skipped. The gpt-5.5 runs use a seeded uniform sample of 400 test questions (`test_sample400_ids.json`, seed 0, built by `python -m ledgermind.data.sample`).

| File prefix | Model | Set | Mode | Cost |
|---|---|---|---|---|
| `gpt54mini_test_pipeline` | gpt-5.4-mini-2026-03-17 | test | pipeline v1 | $2.03 |
| `gpt54mini_test_direct_{default,stated}` | gpt-5.4-mini-2026-03-17 | test | direct | $1.25 |
| `gpt55_sample400_pipeline` | gpt-5.5-2026-04-23 | 400 sample | pipeline v1 | $7.44 |
| `gpt55_sample400_direct_{default,stated}` | gpt-5.5-2026-04-23 | 400 sample | direct | $5.16 |

Raw outputs are the `gpt-5.*.jsonl.gz` files. Direct answers are scored two ways: `default` with the same matching as every other run, and `stated` with `--stated-precision`, which accepts an answer that equals the gold value rounded to the decimals the model wrote (at least two significant digits) and evaluates fraction answers. The stated rule was added after reading the first direct outputs, where correct rounded answers were marked wrong, so both scores are reported.

## Open weight baselines (`open/`)

Self hosted on the cluster with vLLM 0.31 and a reasoning parser that strips the thinking from each reply (`slurm/eval_open.sbatch`), greedy decoding, batch invariant mode off, up to 12,288 output tokens and a 16,384 token context. One job per model runs five passes: test through pipeline prompt v1 and v2, test as a direct answer, and counterfactual test through pipeline v2 and directly. Direct answers are rescored with `--stated-precision` locally (the `_stated` files). `systems.json` lists which files the dashboard uses.

| Folder | Model | GPU | Job | Code |
|---|---|---|---|---|
| `gptoss120b/` | openai/gpt-oss-120b (MXFP4, Marlin MoE kernels, reasoning effort default) | A100 80GB | 10972645 | 417cf40 |

These runs used A100 GPUs while the trained models ran on H200, so their latency and throughput are not comparable and they are left out of the cost chart.

## Paired comparisons (`compare/`)

Produced by `python -m ledgermind.eval.compare` from the rows files above (McNemar and paired bootstrap on the same questions).

| File | A vs B |
|---|---|
| `compare_sft_vs_base_correct_{scale,strict}.json` | `sft_bi1_test` vs `base_zeroshot_bi_test` |
| `compare_grpo_vs_sft_correct_{scale,strict}.json` | `grpo_test` vs `sft_bi1_test` |
| `compare_grpo_fp8_vs_bf16_correct_{scale,strict}.json` | `grpo_fp8_test` vs `grpo_test` |
| `compare_grpo_v2_vs_v1_correct_{scale,strict}.json` | `grpo_v2_test` vs `grpo_test` (prompt v2 against v1, same model) |
| `compare_sft_v2_vs_v1_correct_{scale,strict}.json` | `sft_v2_test` vs `sft_bi1_test` (prompt v2 against v1, same model) |
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
