# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-07-06

### Added

- Model caching with thread-safe LRU cache for summarizers and multimodal pipelines
- SSE streaming endpoint `POST /summarize/stream`
- API key auth, rate limiting, and fail-closed `TRAIN_API_KEY` with `REQUIRE_TRAIN_KEY=1`
- Tier-4 G-Eval evaluation (optional `deepeval` in `[eval]` extras)
- Benchmark script (`scripts/run_benchmarks.py`) and human eval template
- Citation spans pipeline (`pipelines/citations.py`)
- Multi-document RAG summarization
- YAML-defined custom rubrics in `config/rubrics/`
- Video scene detection via ffmpeg scene filter and chapter markers in output
- LangChain tool adapter (`integrations/langchain_tools.py`)
- OpenAPI export script, MCP plugin docs, deployment guide
- Docker Compose setup with optional GPU profile
- Colab demo notebook, nightly CI workflow, GitHub Release workflow
- Good first issue template

### Changed

- API version bumped to 1.2.0
- Video keyframe extraction uses scene-change detection instead of fixed I-frame sampling

## [0.1.0] - 2026-07-06

### Added

- Modern `src/textSummarizer` package layout with `pyproject.toml` and `uv`
- Multi-model registry: Pegasus, BART, T5, FLAN-T5, LongT5, extractive
- Long-document strategies: stuff, map-reduce, refine
- Multi-metric evaluation suite (ROUGE, optional BERTScore/SummaC)
- FastAPI serving with Pydantic request/response models
- Gradio HuggingFace Space demo
- CI with ruff and pytest
- Contributor infrastructure: CONTRIBUTING, issue templates, CODEOWNERS

### Fixed

- Broken package imports from flat layout
- Missing `config.yaml` and `params.yaml`
- `eval_steps` bug in configuration manager
- Data validation logic
- Removed `pipeline("summarization")` dependency

### Changed

- Training hyperparameters read from `params.yaml`
- Evaluation uses configurable sample count and JSON output
