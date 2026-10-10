#!/bin/bash
# Score an open weight baseline model with one vLLM server for all passes: on test through
# the pipeline with prompt v1 (the prompt the API baselines saw) and v2 (conventions spelled
# out), and as a direct answer; on counterfactual test through pipeline v2 and directly.
#
#   sbatch slurm/eval_open.sbatch <hub_id> <name> <reasoning_parser>
#
# These models reason before answering, so the server gets a long context and a reasoning
# parser that strips the thinking from the reply. Batch invariant mode is off because its
# kernels do not cover every architecture; each pass is a single greedy run.
#SBATCH --job-name=lm-evalopen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

MODEL="${1:?hub id}"
NAME="${2:?run name}"
PARSER="${3:?reasoning parser, for example qwen3 or openai_gptoss}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_EXTRA_ARGS="--reasoning-parser $PARSER ${VLLM_EXTRA_ARGS:-}"
export VLLM_BATCH_INVARIANT=0
mkdir -p outputs/eval

echo "model $MODEL name $NAME parser $PARSER max_tokens $MAX_TOKENS max_model_len $MAX_MODEL_LEN git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
serving/launch_vllm.sh "$MODEL" bf16 "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM ready on :$PORT"

# split:mode where mode is pipeline_v1, pipeline_v2 or direct
for PASS in test:pipeline_v1 test:pipeline_v2 test:direct \
            counterfactual_test:pipeline_v2 counterfactual_test:direct; do
  SPLIT=${PASS%%:*}; MODE=${PASS#*:}
  if [ "$MODE" = direct ]; then FLAGS="--direct"; else FLAGS="--prompt-version ${MODE#pipeline_}"; fi
  SCORE=""; [ "$MODE" = direct ] && SCORE="--direct"
  OUT="outputs/eval/${NAME}_${SPLIT}_${MODE}"
  EXAMPLES="data/processed/finqa/$SPLIT.jsonl"
  # shellcheck disable=SC2086
  python -m ledgermind.serving.predict --examples "$EXAMPLES" --endpoint chat $FLAGS \
    --max-tokens "$MAX_TOKENS" --concurrency 64 \
    --base-url "http://localhost:$PORT/v1" --model ledgermind --out "$OUT.jsonl"
  # shellcheck disable=SC2086
  python -m ledgermind.eval.harness --examples "$EXAMPLES" --predictions "$OUT.jsonl" $SCORE \
    --name "${NAME}_${MODE}" --out "$OUT.json" --rows "${OUT}_rows.jsonl"
done
echo EVAL_OPEN_DONE
