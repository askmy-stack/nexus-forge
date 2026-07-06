# Nexus Forge (SummarizeHub)

> **Multimodal summarization platform** — summarize text, images, audio, and video with transformer models, subjective LLM grading, MCP agent integration, and a FastAPI serving layer.

[![CI](https://github.com/askmy-stack/nexus-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/askmy-stack/nexus-forge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![v1.0.0](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/askmy-stack/nexus-forge/releases/tag/v1.0.0)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

![SummarizeHub Demo](docs/assets/demo.gif)

**Nexus Forge** (SummarizeHub) is a production-ready NLP platform for multimodal summarization. Use it as a library, CLI, REST API, MCP server for AI agents, or [HuggingFace Space](https://huggingface.co/spaces) demo. One API surface across extractive and abstractive models, with a grading loop for quality-driven refinement.

---

## Features

- **Four modalities** — text, image (BLIP captioning), audio (Whisper ASR), video (ffmpeg + ASR + keyframe captions)
- **Multi-model registry** — Pegasus, BART, T5, FLAN-T5, LongT5, extractive TextRank-style ranking
- **Long-document strategies** — stuff, map-reduce, refine, hierarchical (RAPTOR), and RAG retrieval
- **MCP server** — 6 tools: `summarize_text`, `summarize_image`, `summarize_audio`, `summarize_video`, `list_models`, `grade_summary`
- **Grading loop** — subjective rubric (coherence, faithfulness, fluency, relevance) with summarize → grade → refine
- **FastAPI serving** — `/summarize`, `/summarize/multimodal`, `/grade`, `/models`, `/train`
- **5-stage training pipeline** — ingest → validate → transform → train → evaluate
- **Cursor skill** — `skills/summarizehub/SKILL.md` for agent integration

---

## Architecture

```mermaid
flowchart LR
    subgraph Input["Multimodal Input"]
        TXT[Text]
        IMG[Image]
        AUD[Audio]
        VID[Video]
    end

    ROUTER[Multimodal Router]
    REG[Model Registry]
    SUM[Summarize]
    STRAT[Strategy Router]
    GRADE[Grade Rubric]
    REFINE[Refine Loop]

    TXT --> ROUTER
    IMG --> ROUTER
    AUD --> ROUTER
    VID --> ROUTER
    ROUTER --> REG --> STRAT --> SUM --> GRADE
    GRADE -->|score < threshold| REFINE --> SUM
    GRADE -->|pass| OUT[Summary]
```

Clients: **CLI** · **FastAPI** · **MCP** · **Gradio Space** · **Cursor agents**

---

## Modalities

| Modality | Pipeline | Default Model | Optional Deps |
|----------|----------|---------------|---------------|
| **Text** | Direct summarization | `extractive` | — |
| **Image** | BLIP caption → summarize | `Salesforce/blip-image-captioning-base` | `pillow` |
| **Audio** | Whisper ASR → summarize | `openai/whisper-tiny` | `soundfile` |
| **Video** | ffmpeg audio + keyframes → Whisper + BLIP → merge | `openai/whisper-tiny` + BLIP | `ffmpeg`, `pillow`, `soundfile` |

---

## Quick start

```bash
# Install from PyPI (or editable from source)
pip install nexus-forge

git clone https://github.com/askmy-stack/nexus-forge.git
cd nexus-forge
uv sync --group dev

# CLI — summarize text (no GPU, extractive model)
uv run text-summarizer \
  --text "AI is transforming industries. Machine learning enables automation." \
  --model extractive

# List registered models
uv run text-summarizer --list-models

# Start API server
uv run uvicorn textSummarizer.serving.app:app --reload --port 8080

# Start MCP server (for AI agents)
uv sync --extra mcp
uv run python -m textSummarizer.mcp.server
```

---

## MCP setup

Add to Cursor `mcp.json`:

```json
{
  "mcpServers": {
    "summarizehub": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/nexus-forge",
        "python",
        "-m",
        "textSummarizer.mcp.server"
      ]
    }
  }
}
```

| Tool | Description |
|------|-------------|
| `summarize_text` | Summarize plain text |
| `summarize_image` | Caption image with BLIP, then summarize |
| `summarize_audio` | Transcribe with Whisper, then summarize |
| `summarize_video` | Extract audio/keyframes, merge ASR + captions, summarize |
| `list_models` | List available summarization models |
| `grade_summary` | Subjective rubric scoring (coherence, faithfulness, fluency, relevance) |

See [skills/summarizehub/SKILL.md](skills/summarizehub/SKILL.md) for agent integration guidance.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and model count |
| `GET` | `/models` | List registered models |
| `POST` | `/summarize` | Summarize text |
| `POST` | `/summarize/multimodal` | Multimodal summarization (JSON + base64) |
| `POST` | `/summarize/multimodal/upload` | Multimodal file upload (image/audio/video) |
| `POST` | `/grade` | Grade a summary against source |
| `POST` | `/train` | Run full training pipeline (requires `TRAIN_API_KEY`) |
| `GET` | `/docs` | OpenAPI interactive docs |

```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "AI is reshaping healthcare.", "model": "extractive", "max_length": 128}'
```

---

## Grading loop

Subjective scoring for loop engineering — no OpenAI API key required (heuristic judge by default):

| Dimension | Scale |
|-----------|-------|
| Coherence | 1–5 |
| Faithfulness | 1–5 |
| Fluency | 1–5 |
| Relevance | 1–5 |

**Flow:** summarize → grade → refine (up to 2 iterations if score < threshold).

```python
from textSummarizer.grading import SummarizationLoop

loop = SummarizationLoop(model="extractive", max_iterations=2)
result = loop.run("Long source text here...", max_length=128)
print(result.score.to_dict())
```

---

## Training pipeline

Five-stage MLOps pipeline orchestrated via CLI or `POST /train`:

| Stage | Module | Purpose |
|-------|--------|---------|
| 1. Ingest | `stage_01_data_ingestion` | Download and load datasets |
| 2. Validate | `stage_02_data_validation` | Schema and quality checks |
| 3. Transform | `stage_03_data_transformation` | Tokenize and split |
| 4. Train | `stage_04_model_trainer` | Fine-tune summarization models |
| 5. Evaluate | `stage_05_model_evaluation` | ROUGE / BERTScore metrics |

```bash
uv run python scripts/run_pipeline.py
```

---

## Project structure

```
src/textSummarizer/
├── components/     # Pipeline stage implementations
├── models/         # Multi-model registry + summarizers
├── pipelines/      # Long-doc strategies (map-reduce, refine, chunking)
├── multimodal/     # Image, audio, video, router
├── grading/        # Rubric, LLM judge, improvement loop
├── mcp/            # MCP server for AI agent integration
├── evaluation/     # Metric suite (ROUGE, BERTScore, SummaC)
├── serving/        # FastAPI app
└── pipeline/       # Stage orchestrators

skills/summarizehub/  # Cursor skill for agent integration
spaces/               # HuggingFace Gradio Space
scripts/              # demo.py, run_pipeline.py, generate_demo_gif.py
docs/assets/          # Demo GIF and static fallback
```

---

## Optional dependencies

```bash
uv sync --extra multimodal   # image + audio + video
uv sync --extra mcp          # MCP server
uv sync --extra demo         # Gradio Space
uv sync --extra eval         # BERTScore
uv sync --extra onnx         # ONNX export + ORT inference
uv sync --extra rag          # BM25 + sentence-transformers RAG
```

Video requires **ffmpeg** on `PATH` (`brew install ffmpeg` / `apt install ffmpeg`).

---

## ONNX inference

Export BART/T5-family models to ONNX for faster CPU inference:

```bash
uv sync --extra onnx
uv run python scripts/export_onnx.py --model bart --output-dir artifacts/onnx/bart

# Use exported model
python -c "
from textSummarizer.models import ModelFactory
s = ModelFactory.create('bart', onnx_dir='artifacts/onnx/bart')
print(s.summarize('AI is reshaping healthcare.', max_length=64))
"
```

Supported export models: `bart`, `t5`, `flan-t5`, `pegasus`, `pegasus-xsum`, `longt5`.

Publish to PyPI: tag a release (`git tag v1.1.0 && git push origin v1.1.0`) or run `./scripts/publish_pypi.sh --upload` with `PYPI_API_TOKEN` set.

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ | Multimodal summarization (text, image, audio, video) |
| ✅ | MCP server + Cursor skill |
| ✅ | Subjective grading loop |
| ✅ | 5-stage training pipeline |
| ✅ | PyPI publish (`pip install nexus-forge`) |
| ✅ | ONNX export for faster inference |
| ✅ | HuggingFace Space with GPU-backed abstractive models |
| ✅ | Hierarchical and RAG-based summarization strategies |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

```bash
uv run pre-commit install
uv run ruff check .
uv run pytest -m "not gpu and not slow and not network"
```

Regenerate the README demo GIF:

```bash
uv run python scripts/generate_demo_gif.py
```

---

## License

MIT — see [LICENSE](LICENSE).
