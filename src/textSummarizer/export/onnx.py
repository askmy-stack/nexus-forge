"""Export BART/T5-family models to ONNX via Hugging Face Optimum."""

from pathlib import Path

ONNX_SUPPORTED_MODELS = frozenset({"bart", "t5", "flan-t5", "pegasus", "pegasus-xsum", "longt5"})


def export_seq2seq_to_onnx(
    model_name: str,
    output_dir: str | Path,
    *,
    opset: int = 14,
) -> Path:
    """Export a registry model to ONNX format for ORT inference."""
    from textSummarizer.models.registry import MODEL_REGISTRY

    key = model_name.lower().replace("_", "-")
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    if key not in ONNX_SUPPORTED_MODELS:
        supported = ", ".join(sorted(ONNX_SUPPORTED_MODELS))
        raise ValueError(
            f"Model '{model_name}' does not support ONNX export. Supported: {supported}"
        )

    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    spec = MODEL_REGISTRY[key]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    tokenizer.save_pretrained(output_path)

    model = ORTModelForSeq2SeqLM.from_pretrained(
        spec.model_id,
        export=True,
        opset=opset,
    )
    model.save_pretrained(output_path)
    return output_path
