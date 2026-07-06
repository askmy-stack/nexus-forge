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
---

Check the main [repository](https://github.com/askmy-stack/nexus-forge) for full documentation.

## Deploy to Hugging Face Spaces

```bash
# 1. Authenticate (one-time)
hf auth login

# 2. Create the Space (one-time; pick your HF username)
hf repo create summarizehub --type space --space-sdk gradio

# 3. Upload from repo root
hf upload <your-hf-username>/summarizehub spaces/ --repo-type space

# 4. Open the Space URL and verify the Gradio demo loads
```

For Pegasus fine-tuning and model publishing:

```bash
# Requires GPU + HF token
export HF_TOKEN=hf_...
uv run python scripts/run_pipeline.py   # stages 4-5: train + evaluate
hf upload <your-hf-username>/pegasus-samsum artifacts/model_trainer/pegasus-samsum-model --repo-type model
```
