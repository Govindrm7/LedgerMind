#!/bin/bash
# Run a CPU only step of the pipeline with the core package (no GPU environment needed).
#
#   sbatch slurm/cpu.sbatch -m ledgermind.saboteur.counterfactual --split test
#SBATCH --job-name=lm-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"

VENV="$LOCAL/core-venv"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -e .
echo "python $* git $(git rev-parse --short HEAD)"
"$VENV/bin/python" "$@"
echo CPU_DONE
