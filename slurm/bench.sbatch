#!/bin/bash
# Latency and throughput sweep for one served variant, in vLLM's default (production) mode.
#
#   sbatch slurm/bench.sbatch <model_path_or_hub_id> <label> [variant]
#
# Runs the concurrency sweep unconstrained and with JSON schema decoding, and writes
# outputs/bench/<label>.json and <label>_constrained.json.
#SBATCH --job-name=lm-bench
#SBATCH --partition=gpu,gpu-short   # short jobs backfill sooner on gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

MODEL="${1:?model path or hub id}"
LABEL="${2:?label}"
VARIANT="${3:-bf16}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
mkdir -p outputs/bench

echo "model $MODEL label $LABEL variant $VARIANT git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
serving/launch_vllm.sh "$MODEL" "$VARIANT" "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
echo "vLLM ready on :$PORT"

URL="http://localhost:$PORT/v1"
EXAMPLES=data/processed/finqa/test.jsonl
# Warm up compiled graphs and caches so the first sweep level is not penalized.
python -m ledgermind.serving.bench --base-url "$URL" --model ledgermind --examples "$EXAMPLES" \
  --concurrency 16 --requests 64 --label warmup --out "outputs/bench/${LABEL}_warmup.json" >/dev/null
python -m ledgermind.serving.bench --base-url "$URL" --model ledgermind --examples "$EXAMPLES" \
  --label "$LABEL" --out "outputs/bench/$LABEL.json"
python -m ledgermind.serving.bench --base-url "$URL" --model ledgermind --examples "$EXAMPLES" \
  --constrained --label "${LABEL}_constrained" --out "outputs/bench/${LABEL}_constrained.json"
echo BENCH_DONE
