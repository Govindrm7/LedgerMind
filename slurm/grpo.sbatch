#!/bin/bash
# Stage 2: GRPO with verifier rewards, then merge the adapter for serving.
#
#   sbatch slurm/grpo.sbatch [config]
#
# Runs past the 8 hour limit by chaining: the trainer resumes from the newest checkpoint,
# and finished stages are skipped, so a second job can pick up where the first stopped:
#   first=$(sbatch --parsable slurm/grpo.sbatch)
#   sbatch --dependency=afterany:$first slurm/grpo.sbatch
#SBATCH --job-name=lm-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

CONFIG="${1:-training/configs/grpo_qwen3_4b.yaml}"
BASE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model'])" "$CONFIG")
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "config $CONFIG base $BASE out $OUT git $(git rev-parse --short HEAD)"

if [ -f "$OUT/run.json" ]; then
  echo "training already finished ($OUT/run.json)"
else
  python training/grpo.py --config "$CONFIG"
fi
if [ -f "${OUT}_merged/config.json" ]; then
  echo "already merged (${OUT}_merged)"
else
  python training/merge_adapter.py --base "$BASE" --adapter "$OUT" --out "${OUT}_merged"
fi
echo GRPO_DONE
