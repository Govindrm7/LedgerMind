#!/bin/bash
# Stage 1: SFT with LoRA, then merge the adapter into the base weights for vLLM.
#
#   sbatch slurm/sft.sbatch [config]
#SBATCH --job-name=lm-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

CONFIG="${1:-training/configs/sft_qwen3_4b.yaml}"
BASE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model'])" "$CONFIG")
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "config $CONFIG base $BASE out $OUT git $(git rev-parse --short HEAD)"

python training/sft.py --config "$CONFIG"
python training/merge_adapter.py --base "$BASE" --adapter "$OUT" --out "${OUT}_merged"
echo SFT_DONE
