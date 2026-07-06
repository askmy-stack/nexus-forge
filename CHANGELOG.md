# Changelog

All notable changes to this project will be documented in this file.

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
