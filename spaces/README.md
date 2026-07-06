---
title: SummarizeHub
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
hardware: zero-gpu
---

GPU-backed abstractive summarization demo for [Nexus Forge](https://github.com/askmy-stack/nexus-forge).

## Features

- Default model: **BART** (`facebook/bart-large-cnn`) on GPU via `@spaces.GPU`
- Models: BART, FLAN-T5, T5, Pegasus, extractive
- Strategies: stuff, map_reduce, refine, hierarchical, rag

## Deploy to Hugging Face Spaces

```bash
# 1. Authenticate (one-time)
hf auth login

# 2. Deploy from repo root (creates Space if missing)
export HF_USERNAME=your-hf-username
export HF_SPACE_NAME=summarizehub   # optional, default: summarizehub
./scripts/deploy_hf_space.sh

# Or manually:
hf repo create summarizehub --type space --space-sdk gradio --space-hardware zero-gpu
hf upload <username>/summarizehub spaces/ --repo-type space
```

Enable **ZeroGPU** in Space settings for free GPU bursts. The `@spaces.GPU` decorator in `app.py` requests GPU only during inference.

## Local development

```bash
cd spaces
pip install -r requirements.txt
python app.py
```

For full repo docs see the [main README](https://github.com/askmy-stack/nexus-forge).
