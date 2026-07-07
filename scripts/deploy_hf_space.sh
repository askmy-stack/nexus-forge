#!/usr/bin/env bash
# Deploy the Gradio Space to Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPACE_NAME="${HF_SPACE_NAME:-summarizehub}"
HF_USER="${HF_USERNAME:-}"

if [[ -z "$HF_USER" ]]; then
  if command -v hf >/dev/null 2>&1; then
    HF_USER="$(hf auth whoami 2>/dev/null | head -1 || true)"
  fi
fi

if [[ -z "$HF_USER" ]]; then
  echo "Set HF_USERNAME or run: hf auth login" >&2
  echo ""
  echo "Manual deploy:"
  echo "  hf repo create ${SPACE_NAME} --type space --space-sdk gradio --flavor cpu-basic"
  echo "  hf upload <username>/${SPACE_NAME} spaces/ --repo-type space"
  exit 1
fi

REPO="${HF_USER}/${SPACE_NAME}"
echo "==> Deploying Space: ${REPO}"

if ! hf repo info "$REPO" --repo-type space >/dev/null 2>&1; then
  echo "==> Creating Space (cpu-basic)"
  hf repo create "$SPACE_NAME" --type space --space-sdk gradio --flavor cpu-basic
fi

hf upload "$REPO" spaces/ --repo-type space
echo "==> Done: https://huggingface.co/spaces/${REPO}"
