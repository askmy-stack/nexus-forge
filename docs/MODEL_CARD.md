---
language: en
license: mit
tags:
  - summarization
  - text2text-generation
  - pegasus
datasets:
  - samsum
metrics:
  - rouge
  - bertscore
library_name: transformers
pipeline_tag: summarization
---

# pegasus-samsum-model

Fine-tuned Pegasus model for dialogue summarization on the SAMSum dataset.

## Usage

```python
from textSummarizer.models import ModelFactory

summarizer = ModelFactory.create("pegasus", model_path="path/to/model")
summary = summarizer.summarize("Your dialogue here...")
```

## Evaluation

| Metric | Score |
|--------|-------|
| ROUGE-1 | TBD |
| ROUGE-2 | TBD |
| ROUGE-L | TBD |

Run `uv run python scripts/run_pipeline.py` to reproduce training and evaluation.
