"""CLI for text summarization."""

import argparse

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.models import ModelFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize text")
    parser.add_argument("--text", required=True, help="Input text to summarize")
    parser.add_argument("--model", default="extractive", help="Model name")
    parser.add_argument("--strategy", default="stuff", choices=["stuff", "map_reduce", "refine"])
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.model == "pegasus":
        pipeline = PredictionPipeline(model_name="pegasus")
        summary = pipeline.predict(args.text, strategy=args.strategy, max_length=args.max_length)
    else:
        summarizer = ModelFactory.create(args.model)
        summary = summarizer.summarize(
            args.text, max_length=args.max_length, strategy=args.strategy
        )

    print(summary)


if __name__ == "__main__":
    main()
