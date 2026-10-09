#!/bin/bash
# Serve a model with vLLM on this node, generate answers for a split, and score them.
#
#   sbatch slurm/eval.sbatch <model_path_or_hub_id> <name> [split] [variant]
#
# Writes outputs/eval/<name>_<split>.jsonl (raw outputs), <name>_<split>.json (metrics)
# and <name>_<split>_rows.jsonl (per question verdicts).
#SBATCH --job-name=lm-eval
#SBATCH --partition=gpu,gpu-short   # short jobs backfill sooner on gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

# Scored runs use batch invariant kernels: greedy decoding then gives identical outputs on
# every run, which paired significance tests need. Latency benchmarks use the default mode.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

MODEL="${1:?model path or hub id}"
NAME="${2:?run name}"
SPLIT="${3:-test}"
VARIANT="${4:-bf16}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
EXAMPLES="data/processed/finqa/$SPLIT.jsonl"
OUT="outputs/eval/${NAME}_${SPLIT}"
mkdir -p outputs/eval

echo "model $MODEL name $NAME split $SPLIT variant $VARIANT batch_invariant $VLLM_BATCH_INVARIANT git $(git rev-parse --short HEAD)"
serving/launch_vllm.sh "$MODEL" "$VARIANT" "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
echo "vLLM ready on :$PORT"

python -m ledgermind.serving.predict --examples "$EXAMPLES" \
  --base-url "http://localhost:$PORT/v1" --model ledgermind --out "$OUT.jsonl"
python -m ledgermind.eval.harness --examples "$EXAMPLES" --predictions "$OUT.jsonl" \
  --name "$NAME" --out "$OUT.json" --rows "${OUT}_rows.jsonl"
cat "$OUT.json"
echo EVAL_DONE
