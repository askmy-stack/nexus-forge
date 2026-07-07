#!/usr/bin/env python3
"""Run summarization benchmarks on a small fixture dataset."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textSummarizer.evaluation.metrics import EvaluationSuite  # noqa: E402
from textSummarizer.models import ModelFactory  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "benchmark_samples.json"
OUTPUT = ROOT / "docs" / "benchmarks.md"


def load_fixtures() -> list[dict]:
    if not FIXTURES.exists():
        return [
            {
                "source": (
                    "Artificial intelligence is transforming healthcare. "
                    "Machine learning detects diseases from medical images."
                ),
                "reference": "AI transforms healthcare with ML-based disease detection.",
            },
            {
                "source": (
                    "Climate change raises global temperatures. "
                    "Renewable energy adoption is accelerating worldwide."
                ),
                "reference": "Climate change and renewable energy adoption are key trends.",
            },
        ]
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def main() -> None:
    samples = load_fixtures()
    summarizer = ModelFactory.create("extractive")
    predictions = [
        summarizer.summarize(sample["source"], max_length=64, strategy="stuff")
        for sample in samples
    ]
    references = [sample["reference"] for sample in samples]
    sources = [sample["source"] for sample in samples]

    suite = EvaluationSuite(tier=4)
    scores = suite.evaluate(predictions, references, sources=sources)

    lines = [
        "# Benchmark Results",
        "",
        f"_Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Configuration",
        "",
        "- Model: `extractive`",
        "- Strategy: `stuff`",
        "- Evaluation tier: 4 (ROUGE, BERTScore, SummaC, G-Eval)",
        f"- Samples: {len(samples)}",
        "",
        "## Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for key, value in sorted(scores.items()):
        lines.append(f"| `{key}` | {value:.4f} |")

    lines.extend(
        [
            "",
            "## Sample Outputs",
            "",
        ]
    )
    for index, (sample, prediction) in enumerate(zip(samples, predictions, strict=True), start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                f"**Source:** {sample['source']}",
                "",
                f"**Prediction:** {prediction}",
                "",
                f"**Reference:** {sample['reference']}",
                "",
            ]
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
