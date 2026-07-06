---
name: summarizehub
description: >-
  Summarize text, images, audio, and voice using SummarizeHub — extractive/abstractive
  models, long-document strategies, multimodal caption/transcribe pipelines, subjective
  grading, and MCP agent tools. Use when the user asks to summarize content, grade
  summaries, use SummarizeHub MCP, or work with the nexus-forge platform.
---

# SummarizeHub

Production-ready multimodal summarization platform at `nexus-forge`. Agents should prefer SummarizeHub's registry, strategies, and MCP tools over ad-hoc `transformers.pipeline("summarization")` calls.

## When to use

**Use this skill when:**
- Summarizing text, articles, transcripts, long documents, or videos
- Summarizing images (caption → summarize), audio/voice (transcribe → summarize), or video (ASR + keyframes → summarize)
- Grading summary quality with a subjective rubric
- Integrating summarization into agents via MCP or REST
- Choosing models/strategies for extractive vs abstractive trade-offs

**Do NOT use when:**
- General NLP unrelated to summarization (NER, translation, sentiment)
- The user wants a different summarization library (LangChain chains, raw OpenAI API)
- Video summarization without ffmpeg installed (install ffmpeg first)
- Training custom models from scratch without the SummarizeHub pipeline

## Prerequisites

| Requirement | Command / notes |
|-------------|-----------------|
| Python 3.11+ | `requires-python = ">=3.11,<3.14"` |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| Base install | `uv sync` in repo root |
| Multimodal (image/audio/video) | `uv pip install -e ".[multimodal]"` — adds Pillow, soundfile |
| Video extraction | [ffmpeg](https://ffmpeg.org/download.html) on PATH (system dependency) |
| MCP server | `uv pip install -e ".[mcp]"` — adds `mcp>=1.0` |
| Full agent stack | `uv pip install -e ".[multimodal,mcp]"` |
| GPU (optional) | Speeds abstractive models; `extractive` works CPU-only |

Repo: `https://github.com/askmy-stack/nexus-forge`

## Quick start

### CLI (text only)

```bash
cd nexus-forge
uv sync
curl http://localhost:8080/models
uv run text-summarizer summarize \
  "AI is transforming industries. Machine learning enables automation." \
  model extractive strategy map_reduce
```

### REST API

```bash
uv run uvicorn textSummarizer.serving.app:app -p 8080
curl http://localhost:8080/health
```

### MCP server

```bash
uv pip install -e ".[multimodal,mcp]"
uv run summarizehub-mcp
# or: uv run python -m textSummarizer.mcp.server
```

Add to Cursor MCP config — see [MCP integration](#mcp-integration) or [references/mcp-config.json](references/mcp-config.json).

## Multimodal workflows

### Text summarization

1. Pick a model from the registry (default: `extractive` for CPU/no-GPU).
2. Pick a strategy for long documents:
   - `stuff` — single pass (short docs)
   - `map_reduce` — chunk, summarize each, merge
   - `refine` — iterative refinement (abstractive only)
3. Call via CLI, REST `/summarize`, or MCP `summarize_text`.

| Model | Type | Max tokens | Best for |
|-------|------|------------|----------|
| `extractive` | Extractive | 10K | Fast baseline, no GPU |
| `bart` | Abstractive | 1024 | News articles |
| `t5` / `flan-t5` | Abstractive | 512 | Instruction-style |
| `pegasus` / `pegasus-xsum` | Abstractive | 1024 | Articles / extreme abstractive |
| `longt5` | Abstractive | 16K | Long documents |

### Image → caption → summarize

Pipeline: BLIP (`Salesforce/blip-image-captioning-base`) captions the image, then a text model summarizes the caption.

```python
from textSummarizer.multimodal.router import MultimodalRouter
from textSummarizer.multimodal.base import InputType, MultimodalInput

router = MultimodalRouter(text_model="extractive")
result = router.summarize(
    MultimodalInput(input_type=InputType.IMAGE, path="/path/to/image.png"),
    max_length=128,
)
# result: {input_type, caption, summary, model, strategy}
```

MCP: `summarize_image(path="...", model="extractive")` or `base64_data="..."`.

### Audio/voice → transcribe → summarize

Pipeline: Whisper (`openai/whisper-tiny` default) transcribes, then text model summarizes.

```python
result = router.summarize(
    MultimodalInput(input_type=InputType.AUDIO, path="/path/to/audio.wav"),
)
# result: {input_type, transcript, summary, model, strategy}
```

MCP: `summarize_audio(path="...", model="extractive")` or `base64_data="..."`.

### Video → ffmpeg → ASR + keyframes → summarize

Pipeline: ffmpeg extracts 16 kHz mono audio and up to 20 keyframes (1 fps), Whisper transcribes with timestamps, BLIP captions frames, content is merged into a timestamped document, then `map_reduce` summarizes.

```python
result = router.summarize(
    MultimodalInput(input_type=InputType.VIDEO, path="/path/to/video.mp4"),
    strategy="map_reduce",
)
# result: {input_type, document, transcript, visual_captions, summary, model, strategy}
```

MCP: `summarize_video(path="...", model="extractive", strategy="map_reduce")` or `base64_data="..."`.

Requires **ffmpeg** on PATH. Video uploads via REST are limited to 50 MB.

## MCP integration

### Cursor `mcp.json` snippet

Replace `REPO_PATH` with the absolute path to the repo (e.g. `nexus-forge`):

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "bash",
      "args": [
        "-c",
        "cd REPO_PATH && uv run summarizehub-mcp"
      ]
    }
  }
}
```

Copy-ready version: [references/mcp-config.json](references/mcp-config.json).

### Tool reference

All tools return JSON strings. Parse with `json.loads()` before use.

| Tool | Parameters | Returns |
|------|------------|---------|
| `summarize_text` | `text` (required), `model` (default `extractive`), `strategy` (default `stuff`), `max_length` (default `128`) | `{summary, model, strategy}` |
| `summarize_image` | `path` or `base64_data`, `model`, `max_length` | `{input_type, caption, summary, model, ...}` |
| `summarize_audio` | `path` or `base64_data`, `model`, `max_length` | `{input_type, transcript, summary, model, ...}` |
| `summarize_video` | `path` or `base64_data`, `model`, `max_length`, `strategy` (default `map_reduce`) | `{input_type, document, transcript, visual_captions, summary, model, ...}` |
| `grade_summary` | `source`, `summary`, `threshold` (default `3.5`) | `{score, passes, threshold}` |
| `list_models` | (none) | `{model_name: description, ...}` |

### Agent usage pattern

1. Call `list_models` to discover available models.
2. Call the appropriate summarize tool for the modality.
3. Optionally call `grade_summary` with source text and produced summary.
4. If `passes` is false, refine and re-summarize (see grading loop).

## Grading loop

SummarizeHub grades summaries on four subjective dimensions (1–5 scale):

| Dimension | What it measures |
|-----------|------------------|
| `coherence` | Logical flow and internal consistency |
| `faithfulness` | Factual alignment with source |
| `fluency` | Grammar, readability, natural phrasing |
| `relevance` | Coverage of key source content |

Default pass threshold: **3.5** average.

### Loop engineering pattern

```
summarize → grade → (if fails) refine → re-summarize → grade again
```

Example agent loop:

1. `summary = summarize_text(text=source, model="bart", strategy="map_reduce")`
2. `grade = grade_summary(source=source, summary=summary.summary, threshold=3.5)`
3. If `grade.passes` is false, read `grade.score.feedback` and:
   - Retry with a different model (`longt5` for long docs)
   - Switch strategy (`refine` for fluency)
   - Increase `max_length` if relevance is low
4. Stop after pass or max iterations (recommend 3).

Default judge uses heuristics (`LLMJudge(use_llm=False)`). Set `use_llm=True` in code for FLAN-T5 refinement feedback.

Programmatic loop helper:

```python
from textSummarizer.grading.loop import SummarizationLoop

result = SummarizationLoop(model="extractive", max_iterations=2).run(
    source=article, max_length=128, strategy="stuff"
)
# result.summary, result.score, result.iterations, result.history
```

## API reference

Base URL: `http://localhost:8080` (default uvicorn port).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health |
| `GET` | `/models` | Model registry |
| `POST` | `/summarize` | Text summarization |
| `POST` | `/summarize/multimodal` | JSON multimodal (text/image/audio/video) |
| `POST` | `/summarize/multimodal/upload` | Multipart file upload (image/audio/video) |
| `POST` | `/grade` | Subjective summary grading |
| `POST` | `/train` | Training pipeline (requires `TRAIN_API_KEY`) |

Detailed curl examples: [references/api-examples.md](references/api-examples.md).

### Request/response shapes

**POST `/summarize`**

```json
{"text": "...", "model": "extractive", "strategy": "stuff", "max_length": 128}
→ {"summary": "...", "model": "extractive", "strategy": "stuff"}
```

**POST `/summarize/multimodal`**

```json
{"input_type": "image", "base64_data": "<base64>", "model": "extractive", "max_length": 128}
→ {"input_type": "image", "summary": "...", "caption": "...", "model": "...", "strategy": "..."}
```

**POST `/grade`**

```json
{"source": "...", "summary": "...", "threshold": 3.5}
→ {"score": {"coherence": 4, "faithfulness": 4, "fluency": 3, "relevance": 4, "average": 3.75, "feedback": "..."}, "passes": true, "threshold": 3.5}
```

## Frontend / agent integration

### Option A: MCP (recommended for Cursor agents)

Configure `mcp.json`, restart Cursor, call tools by name. No HTTP server needed.

### Option B: REST API

Start uvicorn, call endpoints from any frontend or agent framework:

```python
import httpx

resp = httpx.post("http://localhost:8080/summarize", json={
    "text": article, "model": "bart", "strategy": "map_reduce"
})
summary = resp.json()["summary"]
```

### Option C: Python library

```python
from textSummarizer.models import ModelFactory

summarizer = ModelFactory.create("extractive")
summary = summarizer.summarize(text, max_length=128, strategy="map_reduce")
```

### Option D: HuggingFace Space

Gradio demo in `spaces/` — suitable for human-facing UI, not agent automation.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Unknown model 'X'` | Invalid model name | Call `list_models` / `GET /models` |
| `ImportError: multimodal extras` | Missing Pillow/soundfile | `uv pip install -e ".[multimodal]"` |
| MCP server won't start | Missing mcp package | `uv pip install -e ".[mcp]"` |
| Slow first request | Model download/load | Expected; models cache in `~/.cache/huggingface` |
| OOM on GPU | Large abstractive model | Use `extractive` or smaller model (`flan-t5`) |
| `NotImplementedError` for video | Old SummarizeHub versions | Upgrade; install ffmpeg on PATH |
| `ffmpeg is required` | ffmpeg missing | `brew install ffmpeg` or `apt install ffmpeg` |
| Empty image/audio result | Bad input path or encoding | Verify file exists; use base64 with data-URI prefix support |
| `422` on `/summarize` | Invalid strategy | Use `stuff`, `map_reduce`, or `refine` |

Verify MCP: `bash scripts/smoke-test.sh REPO_PATH`

## Anti-patterns

| Don't | Do instead |
|-------|------------|
| `pipeline("summarization", model="...")` directly | `ModelFactory.create(name).summarize(...)` or MCP tools |
| Load all 7 models at startup | Lazy-load via `ModelFactory`; one model per request |
| Use abstractive without GPU in production | Default to `extractive`; upgrade when GPU available |
| Skip grading for user-facing summaries | Run `grade_summary` loop for quality gates |
| Pass raw video without ffmpeg | Install ffmpeg, then `summarize_video` |
| Hardcode model paths | Use registry names from `list_models` |
| Ignore `strategy` for long docs | Use `map_reduce` or `longt5` for >1024 tokens |

## Additional resources

- [references/mcp-config.json](references/mcp-config.json) — copy-ready MCP config
- [references/api-examples.md](references/api-examples.md) — curl examples for all endpoints
- [scripts/smoke-test.sh](scripts/smoke-test.sh) — verify MCP tools register and respond
