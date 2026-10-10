#!/bin/bash
# Download model weights into the Hugging Face cache on scratch before a GPU job needs them.
#
#   sbatch slurm/download.sbatch Qwen/Qwen3.8-27B
#SBATCH --job-name=lm-download
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"

for MODEL in "$@"; do
  echo "downloading $MODEL to $HF_HOME"
  uvx --from "huggingface_hub[hf_xet]" hf download "$MODEL" --max-workers 8
done
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
echo DOWNLOAD_DONE
