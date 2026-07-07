# Deployment Guide

Step-by-step instructions for PyPI publishing, HuggingFace Spaces, and GPU training.

## PyPI Publishing

### Prerequisites

- PyPI account with API token
- `uv` and build tools installed

### Dry run (local)

```bash
./scripts/publish_pypi.sh
```

This builds `dist/` and runs `twine check` without uploading.

### Upload to PyPI

```bash
export PYPI_API_TOKEN=pypi-...
./scripts/publish_pypi.sh --upload
```

### GitHub Release (automated)

Tag a release to trigger `.github/workflows/release.yml`:

```bash
git tag v1.2.0
git push origin v1.2.0
```

Requires `PYPI_API_TOKEN` secret in GitHub repository settings.

## HuggingFace Space Deployment

### Prerequisites

- HuggingFace account
- `hf` CLI authenticated (`hf auth login`)

### Deploy

```bash
export HF_USERNAME=your-username
export HF_SPACE_NAME=summarizehub   # optional, default: summarizehub
./scripts/deploy_hf_space.sh
```

### GPU-backed Space

For abstractive models, upgrade hardware in Space settings or during creation:

```bash
hf repo create summarizehub --type space --space-sdk gradio --space-hardware "t4-small"
```

Set `HF_TOKEN` as a Space secret for gated models.

## GPU Training (Pegasus)

### Local GPU

```bash
# Requires NVIDIA GPU + CUDA drivers
./scripts/train_gpu.sh
```

### Cloud GPU options

| Provider | Notes |
|----------|-------|
| Google Colab | See `notebooks/nexus_forge_demo.ipynb` |
| Lambda Labs | Rent A10/A100, clone repo, run `train_gpu.sh` |
| AWS SageMaker | Use GPU instance, mount data, run pipeline |
| HF Spaces | Training jobs via `train_gpu.sh` on GPU Space |

### Environment

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=...   # optional, for experiment tracking
uv sync --extra mlops
```

### Credential checklist

- [ ] `HF_TOKEN` — model/dataset download
- [ ] `WANDB_API_KEY` — experiment tracking (optional)
- [ ] `PYPI_API_TOKEN` — package publish
- [ ] `HF_USERNAME` — Space deploy

## Production API

```bash
export API_KEY=your-api-key          # protect /summarize, /grade
export TRAIN_API_KEY=train-secret    # protect /train
export REQUIRE_TRAIN_KEY=1           # fail closed if TRAIN_API_KEY unset
export RATE_LIMIT_PER_MINUTE=60

docker compose up api
```

See `docker-compose.yml` for containerized deployment.

## OpenAPI Export

```bash
uv run python scripts/export_openapi.py
# Output: docs/openapi.json
```
