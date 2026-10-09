#!/bin/bash
# Resolve slurm/requirements-cu129.in into a fully pinned lock for Linux x86_64, Python 3.12.
set -euo pipefail
cd "$(dirname "$0")/.."
uv pip compile slurm/requirements-cu129.in \
  --torch-backend cu129 \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --emit-index-url \
  -o slurm/requirements-cu129.lock
