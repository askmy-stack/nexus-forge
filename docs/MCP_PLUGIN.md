# MCP Plugin Installation Guide

SummarizeHub exposes a Model Context Protocol (MCP) server for AI agents. This guide covers setup for Cursor, Claude Desktop, and Windsurf.

## Prerequisites

```bash
git clone https://github.com/askmy-stack/nexus-forge.git
cd nexus-forge
uv sync
uv pip install -e ".[mcp]"
```

Verify the server starts:

```bash
uv run summarizehub-mcp
```

## Cursor

Add to `~/.cursor/mcp.json` (or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "bash",
      "args": [
        "-c",
        "cd /absolute/path/to/nexus-forge && uv run summarizehub-mcp"
      ]
    }
  }
}
```

Restart Cursor. The following tools become available:

- `summarize_text` — text summarization
- `summarize_image` — BLIP caption + summarize
- `summarize_audio` — Whisper ASR + summarize
- `summarize_video` — ffmpeg + ASR + scene keyframes + summarize
- `list_models` — model registry
- `grade_summary` — subjective rubric scoring

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "bash",
      "args": [
        "-c",
        "cd /absolute/path/to/nexus-forge && uv run summarizehub-mcp"
      ]
    }
  }
}
```

On Windows, use the equivalent path under `%APPDATA%\Claude\`.

## Windsurf

Add to Windsurf MCP settings (`.windsurf/mcp.json` or IDE settings):

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "bash",
      "args": [
        "-c",
        "cd /absolute/path/to/nexus-forge && uv run summarizehub-mcp"
      ]
    }
  }
}
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace model download (abstractive models) |
| `OPENAI_API_KEY` | LLM judge grading (optional) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Server won't start | Run `uv sync` and `uv pip install -e ".[mcp]"` |
| Slow first request | Models cache in `~/.cache/huggingface` and in-process LRU |
| Video tools fail | Install `ffmpeg` on PATH |
| Image/audio tools fail | `uv sync --extra multimodal` |

See also [skills/summarizehub/SKILL.md](../skills/summarizehub/SKILL.md).
