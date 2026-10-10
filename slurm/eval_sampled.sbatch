#!/bin/bash
# Score a model with sampled decoding: one server, several seeded passes over a split.
#
#   sbatch slurm/eval_sampled.sbatch <model_path_or_hub_id> <name> [split] [temperature] [seeds]
#
# Writes outputs/eval/<name>_<split>_t<temperature>_s<seed>.{jsonl,json} and _rows.jsonl per
# seed. Batch invariant kernels plus a fixed seed make each pass reproducible.
#SBATCH --job-name=lm-evalsample
#SBATCH --partition=gpu,gpu-short   # short jobs backfill sooner on gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

MODEL="${1:?model path or hub id}"
NAME="${2:?run name}"
SPLIT="${3:-test}"
TEMP="${4:-0.9}"
SEEDS="${5:-0 1 2 3}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
EXAMPLES="data/processed/finqa/$SPLIT.jsonl"
mkdir -p outputs/eval

# A local model must be a complete checkpoint, so this job can safely be chained with
# --dependency=afterany on a training job that ended without a clean exit status.
if [ -d "$MODEL" ] && ! { [ -f "$MODEL/config.json" ] && ls "$MODEL"/*.safetensors >/dev/null 2>&1 \
    && [ -f "$MODEL/tokenizer_config.json" ]; }; then
  # merge_adapter.py writes the tokenizer after the weights, so its presence means the weights finished
  echo "incomplete model at $MODEL (need config.json, safetensors weights, tokenizer_config.json)"; exit 1
fi
echo "model $MODEL name $NAME split $SPLIT temperature $TEMP seeds $SEEDS git $(git rev-parse --short HEAD)"
serving/launch_vllm.sh "$MODEL" bf16 "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
echo "vLLM ready on :$PORT"

for SEED in $SEEDS; do
  OUT="outputs/eval/${NAME}_${SPLIT}_t${TEMP}_s${SEED}"
  python -m ledgermind.serving.predict --examples "$EXAMPLES" --temperature "$TEMP" --seed "$SEED" \
    --base-url "http://localhost:$PORT/v1" --model ledgermind --out "$OUT.jsonl"
  python -m ledgermind.eval.harness --examples "$EXAMPLES" --predictions "$OUT.jsonl" \
    --name "${NAME}_t${TEMP}_s${SEED}" --out "$OUT.json" --rows "${OUT}_rows.jsonl"
done
echo EVAL_SAMPLED_DONE
