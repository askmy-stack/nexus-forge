---
name: summarizehub
description: Multimodal summarization via SummarizeHub MCP tools — text, image, and audio inputs with subjective grading. Use when summarizing documents, images, voice recordings, evaluating summary quality, or configuring the SummarizeHub MCP server.
---

# SummarizeHub

SummarizeHub is a multimodal summarization platform exposed as an MCP server and REST API.

## When to Use

- Summarize plain text (articles, transcripts, notes)
- Caption and summarize images (diagrams, screenshots, photos)
- Transcribe and summarize audio/voice recordings
- List available summarization models
- Grade a summary against its source (subjective rubric scoring)
- Integrate summarization into agent workflows via MCP

## MCP Server Setup

```bash
cd /path/to/nlp-text-summarization
uv sync --extra mcp
uv run summarizehub-mcp
```

### Cursor `mcp.json`

Copy from [references/mcp-config.json](references/mcp-config.json) and replace the `--directory` path.

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

## REST API Reference

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
| `/health` | GET | Service health |

## Multimodal Workflows

### Text-first

1. Call `list_models` to confirm availability.
2. Call `summarize_text` with `model=extractive` for fast CPU results.
3. Use `bart` or `flan-t5` when fluency matters (GPU recommended).

### Image workflow

1. Provide `base64_data` or local `path` to `summarize_image`.
2. Response includes `caption` (BLIP output) and `summary`.
3. Call `grade_summary` with caption as source if validating faithfulness.

### Audio workflow

1. Provide audio via `summarize_audio` (`path` or `base64_data`).
2. Response includes `transcript` (Whisper output) and `summary`.
3. Grade against transcript or original source text.

### Grading loop

1. Summarize with any modality tool.
2. Call `grade_summary` with `source` and `summary`.
3. If `passes` is false, retry with different `model`, `strategy`, or higher `max_length`.
4. For programmatic loops, use `SummarizationLoop` from `textSummarizer.grading`.

## Optional Dependencies

```bash
uv sync --extra multimodal   # Pillow + soundfile for image/audio
uv sync --extra mcp          # MCP server
```

## Anti-Patterns

- **Do not** call `summarize_image` or `summarize_audio` without `uv sync --extra multimodal` — BLIP/Whisper need Pillow/soundfile.
- **Do not** use `video` input_type — returns 422; extract audio frames first.
- **Do not** assume GPU is available — default to `extractive` for CI and agent workflows.
- **Do not** require OpenAI API keys — grading uses local heuristics by default.
- **Do not** pass remote URLs as `path` — MCP tools expect local filesystem paths or `base64_data`.
- **Do not** skip `grade_summary` when quality matters — abstractive models can hallucinate.
- **Do not** use `map_reduce` for short text (<512 tokens) — `stuff` is faster and equivalent.

## Defaults and Constraints

- Models are lazy-loaded on first use.
- `extractive` is the default for speed and CI compatibility.
- Grading threshold default is 3.5 (scale 1–5).
- Max summary length: 16–512 tokens via API.
