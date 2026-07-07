# G-Eval LLM Judge Setup

Nexus Forge supports [deepeval G-Eval](https://docs.confident-ai.com/docs/metrics-llm-evals) for LLM-as-judge summarization quality scoring. When configured, tier-4 evaluation, `POST /grade?use_geval=true`, and the MCP `grade_summary` tool use the full G-Eval pipeline instead of the token-overlap heuristic.

## Install

```bash
uv pip install -e ".[eval]"
```

The `[eval]` extra includes `deepeval` and `bert-score`.

## API keys

Set **one** of:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Default OpenAI key used by deepeval |
| `GEVAL_API_KEY` | Dedicated key for G-Eval (overrides OpenAI when set) |

```bash
export OPENAI_API_KEY=sk-...
# or
export GEVAL_API_KEY=sk-...
```

## Judge model

Configure the evaluation model with `GEVAL_MODEL` (default: `gpt-4o-mini`):

```bash
export GEVAL_MODEL=gpt-4o-mini
```

## Usage

### Evaluation suite (tier 4)

```python
from textSummarizer.evaluation.metrics import EvaluationSuite

suite = EvaluationSuite(tier=4)
scores = suite.evaluate(
    predictions=["AI is reshaping healthcare."],
    references=["AI changes healthcare."],
    sources=["Artificial intelligence is transforming healthcare systems worldwide."],
)
print(scores["geval_score"], scores.get("geval_method"))
```

### REST API

```bash
curl -X POST http://localhost:8080/grade \
  -H "Content-Type: application/json" \
  -d '{
    "source": "AI is transforming healthcare.",
    "summary": "AI transforms healthcare.",
    "use_geval": true
  }'
```

Response includes a `geval` object with `geval_score`, `geval_reason`, `method`, and `model` when `use_geval` is true.

### MCP

The `grade_summary` tool automatically runs G-Eval when keys are configured. Pass `use_geval=false` to force the heuristic rubric only.

## Fallback behavior

Without `deepeval` or an API key, G-Eval falls back to a **heuristic token-overlap score** (method: `heuristic`). This keeps CI and local development free of paid API calls.

Force heuristic mode in code:

```python
from textSummarizer.grading.geval import geval_score

result = geval_score(source, summary, use_geval=False)
```

## Cost notes

Each G-Eval call invokes the configured judge model once per source/summary pair. Use tier 4 evaluation sparingly in production pipelines, or cache scores for repeated summaries.
