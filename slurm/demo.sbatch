#!/bin/bash
# Live demo: serve the GRPO model with vLLM and the LedgerMind demo page on this node.
#
#   sbatch slurm/demo.sbatch [model] [minutes]
#
# Reach it from a laptop through the login node, for example:
#   ssh -N -L 8899:localhost:<demo port> -J explorer <user>@<node>
# The demo port and node are printed in the job log.
#SBATCH --job-name=lm-demo
#SBATCH --partition=gpu,gpu-short   # short jobs backfill sooner on gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

MODEL="${1:-checkpoints/grpo_qwen3_4b_merged}"
MINUTES="${2:-80}"
VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
DEMO_PORT=$((VLLM_PORT + 1000))

echo "model $MODEL node $(hostname) vllm $VLLM_PORT demo $DEMO_PORT git $(git rev-parse --short HEAD)"
serving/launch_vllm.sh "$MODEL" bf16 "$VLLM_PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER $DEMO 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
echo "vLLM ready on :$VLLM_PORT"
python -m ledgermind.serving.demo_server --vllm "http://localhost:$VLLM_PORT/v1" --port "$DEMO_PORT" &
DEMO=$!
echo "DEMO_READY node $(hostname) port $DEMO_PORT"
sleep $((MINUTES * 60))
echo DEMO_DONE
