#!/usr/bin/env python3
"""CLI to export BART/T5-family models to ONNX."""

import argparse
import sys

from textSummarizer.export.onnx import ONNX_SUPPORTED_MODELS, export_seq2seq_to_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summarization models to ONNX format.")
    parser.add_argument(
        "--model",
        default="bart",
        choices=sorted(ONNX_SUPPORTED_MODELS),
        help="Registry model name to export (default: bart)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/onnx",
        help="Directory for exported ONNX model and tokenizer (default: artifacts/onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    try:
        output = export_seq2seq_to_onnx(args.model, args.output_dir, opset=args.opset)
    except ImportError as exc:
        print(
            "Error: ONNX export requires optional dependencies.\n"
            "Install with: pip install nexus-forge[onnx]",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Exported {args.model} to {output}")


if __name__ == "__main__":
    main()
