#!/bin/bash
# Download model weights into the Hugging Face cache on scratch before a GPU job needs them.
#
#   sbatch slurm/download.sbatch Qwen/Qwen3.8-27B
#   HF_DOWNLOAD_ARGS='--exclude original/* metal/*' sbatch slurm/download.sbatch openai/gpt-oss-120b
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
  # shellcheck disable=SC2086  # HF_DOWNLOAD_ARGS holds several flags
  # Pin Python and the hub version: on the compute nodes uvx otherwise picks the system
  # Python and resolves a 2023 huggingface_hub without the hf command.
  uvx --python 3.12 --from "huggingface_hub[hf_xet]>=1.0" hf download "$MODEL" --max-workers 8 \
    ${HF_DOWNLOAD_ARGS:-}
done
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
echo DOWNLOAD_DONE
