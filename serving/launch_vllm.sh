#!/usr/bin/env bash
# Serve a LedgerMind checkpoint with vLLM on an OpenAI compatible endpoint.
#
#   serving/launch_vllm.sh <model_path_or_hub_id> <variant> [port]
#
# Variants:
#   bf16   full precision weights (baseline)
#   fp16   half precision, for GPUs without bf16 support (V100, T4)
#   fp8    dynamic FP8 weights and activations; real speedups need Hopper (H100) or newer
#   awq    a checkpoint already quantized with AWQ (pass its path as the model)
#
# Optional environment overrides, used for open baseline models that reason before answering:
#   MAX_MODEL_LEN    context length (default 8192)
#   VLLM_EXTRA_ARGS  extra vllm serve flags, for example "--reasoning-parser qwen3"
#
# Serve merged checkpoints (base, base+SFT, base+SFT+GRPO) so every variant is measured on
# the same footing; see training/merge_adapter.py.
set -euo pipefail

MODEL="${1:?model path or hub id}"
VARIANT="${2:-bf16}"
PORT="${3:-8000}"

ARGS=(
  --served-model-name ledgermind
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization 0.90
  --port "$PORT"
  --seed 0
)

case "$VARIANT" in
  bf16) ARGS+=(--dtype bfloat16) ;;
  fp16) ARGS+=(--dtype float16) ;;
  fp8)  ARGS+=(--dtype bfloat16 --quantization fp8) ;;
  awq)  ARGS+=(--quantization awq_marlin) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

echo "serving $MODEL as $VARIANT on :$PORT"
# shellcheck disable=SC2086  # word splitting of the extra flags is intended
exec vllm serve "$MODEL" "${ARGS[@]}" ${VLLM_EXTRA_ARGS:-}
