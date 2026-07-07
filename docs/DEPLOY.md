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

- HuggingFace account ([askhugsai profile](https://huggingface.co/askhugsai))
- `hf` CLI installed (`pip install huggingface_hub` or `uv tool install huggingface_hub`)
- **You must authenticate locally** — we cannot run `hf auth login` or upload without your token:

```bash
hf auth login
```

### Deploy (after login)

Set your username and token, then run the deploy script (it uploads `spaces/` to the Hub; do not run upload commands without a valid token):

```bash
export HF_TOKEN=hf_...
export HF_USERNAME=askhugsai
export HF_SPACE_NAME=nexus-forge   # optional; default in script: summarizehub
./scripts/deploy_hf_space.sh
```

**Live Space:** [https://huggingface.co/spaces/askhugsai/nexus-forge](https://huggingface.co/spaces/askhugsai/nexus-forge)

### Manual create (optional)

If the Space does not exist yet, the script creates it with CPU Basic hardware. Equivalent CLI:

```bash
hf repo create nexus-forge --type space --space-sdk gradio --flavor cpu-basic
hf upload askhugsai/nexus-forge spaces/ --repo-type space
```

### GPU-backed Space

For abstractive models, upgrade hardware in Space settings or during creation:

```bash
hf repo create nexus-forge --type space --space-sdk gradio --flavor t4-small
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

## GPU autoscaling

### HuggingFace ZeroGPU (Spaces)

The Space uses `@spaces.GPU(duration=60)` so GPU is allocated only during inference bursts.

1. Create or open your Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Settings → Hardware → ZeroGPU** — enable ZeroGPU (free GPU bursts)
3. Ensure `spaces/README.md` frontmatter or `spaces/space.yaml` includes `hardware: zero-gpu`
4. For sustained load, upgrade to **T4 small** or larger in Space settings

See [spaces/README.md](../spaces/README.md) for deploy steps.

### Docker Compose GPU profile

```bash
# Single GPU replica (default)
docker compose --profile gpu up api-gpu

# Scale replicas (Swarm-compatible; each replica respects MAX_GPU_JOBS)
MAX_GPU_JOBS=2 MODEL_CACHE_TTL_SECONDS=3600 docker compose --profile gpu up --scale api-gpu=2
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_GPU_JOBS` | `2` | Max concurrent GPU inference jobs per replica |
| `MODEL_CACHE_TTL_SECONDS` | `3600` | Evict cached models after idle TTL |

### Kubernetes

For HPA, GPU node pools, and deployment manifests, see [K8S_AUTOSCALING.md](K8S_AUTOSCALING.md).

## G-Eval LLM judging

See [GEVAL.md](GEVAL.md) for `OPENAI_API_KEY` / `GEVAL_API_KEY` setup and tier-4 evaluation.

## OpenAPI Export

```bash
uv run python scripts/export_openapi.py
# Output: docs/openapi.json
```
