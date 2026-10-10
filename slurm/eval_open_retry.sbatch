#!/bin/bash
# Re-ask an open weight baseline only the questions whose requests failed (for example,
# timed out under heavy batching), with a longer timeout and lower concurrency.
#
#   sbatch slurm/eval_open_retry.sbatch <hub_id> <reasoning_parser> <list_file>
#
# Each line of the list file is "<examples jsonl> <out jsonl> <mode>", mode being
# pipeline_v1, pipeline_v2 or direct. Merge the outputs with scripts/merge_retries.py.
#SBATCH --job-name=lm-evalretry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

MODEL="${1:?hub id}"
PARSER="${2:?reasoning parser}"
LIST="${3:?list file}"
MAX_TOKENS="${MAX_TOKENS:-12288}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export VLLM_EXTRA_ARGS="--reasoning-parser $PARSER ${VLLM_EXTRA_ARGS:-}"
export VLLM_BATCH_INVARIANT=0

echo "model $MODEL parser $PARSER list $LIST git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
serving/launch_vllm.sh "$MODEL" bf16 "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM ready on :$PORT"

while read -r EXAMPLES OUT MODE; do
  [ -z "$EXAMPLES" ] && continue
  if [ "$MODE" = direct ]; then FLAGS="--direct"; else FLAGS="--prompt-version ${MODE#pipeline_}"; fi
  # shellcheck disable=SC2086
  python -m ledgermind.serving.predict --examples "$EXAMPLES" --endpoint chat $FLAGS \
    --max-tokens "$MAX_TOKENS" --concurrency 16 --timeout-s 1800 \
    --base-url "http://localhost:$PORT/v1" --model ledgermind --out "$OUT"
done < "$LIST"
echo EVAL_RETRY_DONE
