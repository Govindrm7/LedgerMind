#!/bin/bash
# Download, verify and convert FinQA. CPU only; needs just the core package.
#SBATCH --job-name=lm-prepare
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"

VENV="$LOCAL/core-venv"
uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -e .
"$VENV/bin/python" -m ledgermind.data.prepare
ls -la data/processed/finqa
echo PREPARE_DONE
