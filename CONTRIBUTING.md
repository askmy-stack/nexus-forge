# Contributing to NLP Text Summarization

Thank you for your interest in contributing! This project welcomes improvements to models, evaluation, pipelines, docs, and tests.

## Development setup

```bash
git clone https://github.com/askmy-stack/nexus-forge.git
cd nexus-forge
uv sync
uv run pre-commit install
```

## Commands

```bash
uv run ruff check .
uv run ruff format .
uv run pytest -m "not gpu and not slow and not network"
uv run python scripts/run_pipeline.py          # full training pipeline
uv run uvicorn textSummarizer.serving.app:app -p 8080
uv run python -c "
from textSummarizer.models import ModelFactory
print(ModelFactory.create('extractive').summarize('Sample text for a quick check.'))
"
```

## Publishing to PyPI

1. Set a PyPI API token: [pypi.org/manage/account/token](https://pypi.org/manage/account/token/)
2. Add `PYPI_API_TOKEN` as a GitHub repository secret (used by `.github/workflows/publish.yml`)
3. Build and verify locally:

```bash
./scripts/publish_pypi.sh          # dry run (build + twine check)
export PYPI_API_TOKEN=pypi-...     # never commit this token
./scripts/publish_pypi.sh          # set upload in script or env for manual upload
```

4. Or push a version tag to trigger CI publish:

```bash
git tag v1.1.0
git push origin v1.1.0
```

## HuggingFace Space deploy

```bash
hf auth login
export HF_USERNAME=your-username
./scripts/deploy_hf_space.sh
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
