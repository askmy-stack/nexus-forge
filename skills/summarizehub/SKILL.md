---
name: summarizehub
description: Multimodal summarization via SummarizeHub MCP tools — text, image, and audio inputs with subjective grading. Use when summarizing documents, images, voice recordings, or evaluating summary quality.
---

# SummarizeHub

SummarizeHub is a multimodal summarization platform exposed as an MCP server and REST API. Use this skill when the user wants to summarize text, images, or audio, list available models, or grade summary quality.

## When to Use

- Summarize plain text (articles, transcripts, notes)
- Caption and summarize images (diagrams, screenshots, photos)
- Transcribe and summarize audio/voice recordings
- List available summarization models
- Grade a summary against its source (subjective rubric scoring)
- Integrate summarization into agent workflows via MCP

## MCP Server Setup

Install with MCP extras:

```bash
uv sync --extra mcp
```

Run the stdio server:

```bash
uv run python -m textSummarizer.mcp.server
```

Or use the CLI entry point:

```bash
uv run summarizehub-mcp
```

### Cursor `mcp.json` Example

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/nlp-text-summarization",
        "python",
        "-m",
        "textSummarizer.mcp.server"
      ]
    }
  }
}
```

## Available MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `summarize_text` | Summarize plain text | `text`, `model` (default: `extractive`), `strategy`, `max_length` |
| `summarize_image` | Caption image + summarize | `path` or `base64_data`, `model`, `max_length` |
| `summarize_audio` | Transcribe audio + summarize | `path` or `base64_data`, `model`, `max_length` |
| `list_models` | List registered models | — |
| `grade_summary` | Subjective rubric scoring | `source`, `summary`, `threshold` |

## Supported Modalities

| Modality | Pipeline | Default Models |
|----------|----------|----------------|
| Text | Direct summarization | `extractive` (fast, CPU) |
| Image | BLIP caption → text summary | `Salesforce/blip-image-captioning-base` |
| Audio | Whisper ASR → text summary | `openai/whisper-tiny` |
| Video | Not yet implemented | — |

## REST API Alternative

If MCP is unavailable, use the FastAPI server:

```bash
uv run uvicorn textSummarizer.serving.app:app --port 8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/summarize` | POST | Text summarization |
| `/summarize/multimodal` | POST | JSON multimodal (text/image/audio via base64) |
| `/summarize/multimodal/upload` | POST | File upload (image/audio) |
| `/grade` | POST | Subjective grading |
| `/models` | GET | List models |

## Agent Integration Patterns

1. **Text-first**: Call `summarize_text` with `model=extractive` for fast results; use `bart` or `flan-t5` when fluency matters.
2. **Image workflow**: Provide `base64_data` or a local `path` to `summarize_image`; returns caption + summary.
3. **Voice workflow**: Provide audio via `summarize_audio`; returns transcript + summary.
4. **Quality loop**: After summarizing, call `grade_summary`; if `passes` is false, retry with a different model or strategy.

## Optional Dependencies

```bash
# Image + audio processing
uv sync --extra multimodal

# MCP server
uv sync --extra mcp
```

## Defaults and Constraints

- **No GPU required** for default path (`extractive` model)
- **No OpenAI API key** required — uses local heuristics for grading
- Models are **lazy-loaded** on first use
- Keep `extractive` as the default for speed and CI compatibility
