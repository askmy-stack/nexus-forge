#!/usr/bin/env bash
# GPU training wrapper for Pegasus fine-tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found. GPU training requires an NVIDIA GPU." >&2
  echo "See docs/DEPLOY.md for cloud GPU options (Colab, Lambda, SageMaker)." >&2
  exit 1
fi

echo "==> GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> Running training pipeline on GPU ${CUDA_VISIBLE_DEVICES}"
uv run python scripts/run_pipeline.py "$@"
