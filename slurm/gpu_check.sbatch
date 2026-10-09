#!/bin/bash
# Check that the GPU, the CUDA build of torch and vLLM generation all work on the node.
# Also downloads the base model into the Hugging Face cache for the training jobs.
#
#   sbatch slurm/gpu_check.sbatch [model]
#SBATCH --job-name=lm-gpucheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LEDGERMIND_ENV:-$HOME/.config/ledgermind/env.sh}"
lm_activate

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import time

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
tflops = 20 * 2 * 8192**3 / (time.time() - start) / 1e12
print(f"bf16 matmul {tflops:.0f} TFLOPS")
PY
python - "${1:-Qwen/Qwen3-4B-Base}" <<'PY'
import sys

import vllm
from vllm import LLM, SamplingParams

print("vllm", vllm.__version__)
llm = LLM(sys.argv[1], max_model_len=2048, gpu_memory_utilization=0.5, seed=0)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
print("generation:", repr(out[0].outputs[0].text))
PY
echo GPU_CHECK_OK
