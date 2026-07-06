#!/usr/bin/env bash
# Verify SummarizeHub MCP tools register and respond.
# Usage: bash scripts/smoke-test.sh [REPO_PATH]
set -euo pipefail

REPO_PATH="${1:-$(cd "$(dirname "$0")/../../../Projects/nlp-text-summarization" 2>/dev/null && pwd || echo "")}"

if [[ -z "$REPO_PATH" || ! -d "$REPO_PATH" ]]; then
  echo "Usage: bash smoke-test.sh /path/to/nlp-text-summarization" >&2
  exit 1
fi

cd "$REPO_PATH"

echo "==> Syncing dependencies (mcp + multimodal)..."
uv sync --extra mcp --extra multimodal --quiet

echo "==> Checking MCP tool registration..."
uv run python - <<'PY'
import json
import sys

try:
    from textSummarizer.mcp import server as mcp_server
except ImportError as exc:
    print(f"FAIL: cannot import MCP server: {exc}", file=sys.stderr)
    sys.exit(1)

tools = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
expected = {
    "summarize_text",
    "summarize_image",
    "summarize_audio",
    "summarize_video",
    "list_models",
    "grade_summary",
}
missing = expected - tools
if missing:
    print(f"FAIL: missing tools: {missing}", file=sys.stderr)
    sys.exit(1)
print(f"OK: registered tools: {sorted(tools)}")
PY

echo "==> Testing summarize_text..."
uv run python - <<'PY'
import json
from textSummarizer.mcp.server import summarize_text, list_models, grade_summary

result = json.loads(summarize_text(
    text="AI is changing industries. Machine learning enables automation.",
    model="extractive",
))
assert result["summary"], "empty summary"
print(f"OK: summary={result['summary'][:60]}...")

models = json.loads(list_models())
assert "extractive" in models
print(f"OK: {len(models)} models available")

grade = json.loads(grade_summary(
    source="AI transforms healthcare.",
    summary="AI transforms healthcare.",
))
assert "passes" in grade and "score" in grade
print(f"OK: grade passes={grade['passes']}, avg={grade['score']['average']}")
PY

echo "==> All smoke tests passed."
