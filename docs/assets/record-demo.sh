#!/usr/bin/env bash
# Record a terminal demo GIF for the README.
# Requires: uv, optional `agg` or `terminalizer` / `vhs` for GIF export.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Starting API in background..."
uv run uvicorn textSummarizer.serving.app:app --port 8080 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
sleep 2

echo "Recording demo commands..."
{
  echo "# Health check"
  curl -s http://localhost:8080/health | python -m json.tool
  echo
  echo "# Summarize (extractive)"
  curl -s -X POST http://localhost:8080/summarize \
    -H "Content-Type: application/json" \
    -d '{"text":"AI is transforming healthcare and finance. Machine learning powers automation.","model":"extractive","strategy":"stuff","max_length":64}' \
    | python -m json.tool
} | tee /tmp/summarize-demo.txt

echo
echo "Demo output saved to /tmp/summarize-demo.txt"
echo "To create docs/assets/demo.gif, use vhs or terminalizer with this script as reference."
