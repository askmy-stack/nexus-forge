# Contributing to NLP Text Summarization

Thank you for your interest in contributing! This project welcomes improvements to models, evaluation, pipelines, docs, and tests.

## Development setup

```bash
git clone https://github.com/askmy-stack/nexus-forge.git
cd nexus-forge
uv sync --group dev
uv run pre-commit install
```

## Commands

```bash
uv run ruff check .
uv run ruff format .
uv run pytest -m "not gpu and not slow and not network"
uv run python scripts/run_pipeline.py          # full training pipeline
uv run uvicorn textSummarizer.serving.app:app --reload --port 8080
uv run python -m textSummarizer.cli --text "..." --model extractive
```

## Contribution areas

| Label | Examples |
|-------|----------|
| `model` | Add new summarization model to registry |
| `evaluation` | Add BERTScore, SummaC, G-Eval metrics |
| `pipeline` | Improve map-reduce or refine strategies |
| `docs` | Tutorials, model comparison notebooks |
| `dataset` | Add cnn_dailymail, xsum, billsum support |

## Pull request process

1. Fork the repo and create a feature branch
2. Add tests for new behavior
3. Run `uv run ruff check .` and `uv run pytest`
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`)
5. Open a PR with a clear description and test plan

## Code style

- Python 3.11+
- `ruff` for linting and formatting
- Type hints encouraged
- Do not use `pipeline("summarization")` — use `model.generate()` wrappers
