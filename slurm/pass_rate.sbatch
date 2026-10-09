#!/bin/bash
# Sample k completions per training prompt from the merged SFT model and keep the prompts
# with 0 < pass rate < 1, the only ones that give GRPO a learning signal.
#
#   sbatch slurm/pass_rate.sbatch [model] [k]
#SBATCH --job-name=lm-passrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

MODEL="${1:-checkpoints/sft_qwen3_4b_merged}"
K="${2:-8}"
echo "model $MODEL k $K git $(git rev-parse --short HEAD)"
python training/pass_rate.py --model "$MODEL" --k "$K" \
  --train-file data/processed/finqa/train.jsonl --out outputs/pass_rates.json
echo PASS_RATE_DONE
