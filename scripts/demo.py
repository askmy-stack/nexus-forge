#!/usr/bin/env python3
"""Quick local demo for text summarization."""

from textSummarizer.models import ModelFactory

SAMPLE_TEXT = """
Artificial intelligence is transforming industries worldwide. Machine learning enables
automation at unprecedented scale, from manufacturing robots to fraud detection in banking.
Natural language processing powers virtual assistants, search engines, and document
summarization tools used by millions of people every day.

In healthcare, AI models assist radiologists by flagging anomalies in medical scans.
Financial institutions use predictive models to assess credit risk and detect suspicious
transactions in real time. Transportation networks optimize routes and enable partial
autonomy in vehicles through computer vision and sensor fusion.

Despite rapid progress, responsible deployment requires attention to bias, transparency,
and human oversight. Organizations that combine strong data practices with clear governance
are best positioned to capture long-term value from AI investments.
""".strip()


def main() -> None:
    print("=" * 60)
    print("Text Summarization Demo")
    print("=" * 60)
    print("\nSource text:\n")
    print(SAMPLE_TEXT)
    print("\n" + "-" * 60)

    for model in ("extractive",):
        print(f"\nModel: {model} | Strategy: stuff")
        summarizer = ModelFactory.create(model)
        summary = summarizer.summarize(SAMPLE_TEXT, max_length=96)
        print(summary)

    print("\n" + "-" * 60)
    print("Try the API: uv run uvicorn textSummarizer.serving.app:app --port 8080")
    print("Open docs:   http://localhost:8080/docs")


if __name__ == "__main__":
    main()
