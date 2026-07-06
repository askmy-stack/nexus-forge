"""CLI for text summarization."""

import argparse
import sys

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.models import ModelFactory

STRATEGIES = ["stuff", "map_reduce", "refine"]


def _list_models() -> None:
    print("Available models:\n")
    for name, description in sorted(ModelFactory.list_models().items()):
        print(f"  {name:<14} {description}")
    print("\nStrategies: stuff, map_reduce, refine")
    print("Example:")
    print('  text-summarizer --text "Your article..." --model extractive --strategy map_reduce')


def _summarize(text: str, model: str, strategy: str, max_length: int) -> str:
    if model == "pegasus":
        pipeline = PredictionPipeline(model_name="pegasus")
        return pipeline.predict(text, strategy=strategy, max_length=max_length)

    summarizer = ModelFactory.create(model)
    return summarizer.summarize(text, max_length=max_length, strategy=strategy)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="text-summarizer",
        description="Summarize text with extractive or abstractive transformer models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  text-summarizer --list-models\n"
            '  text-summarizer --text "AI is transforming industries." --model extractive\n'
            "  text-summarizer --text @article.txt --model bart --strategy map_reduce\n"
        ),
    )
    parser.add_argument("--text", help="Input text to summarize (or path prefixed with @)")
    parser.add_argument(
        "--model",
        default="extractive",
        help="Model name from registry (default: extractive)",
    )
    parser.add_argument(
        "--strategy",
        default="stuff",
        choices=STRATEGIES,
        help="Long-document strategy (default: stuff)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum summary length in tokens/words (default: 128)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List registered models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        _list_models()
        return

    if not args.text:
        parser.error("--text is required unless --list-models is used")

    text = args.text
    if text.startswith("@"):
        with open(text[1:], encoding="utf-8") as handle:
            text = handle.read()

    try:
        summary = _summarize(text, args.model, args.strategy, args.max_length)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(summary)


if __name__ == "__main__":
    main()
