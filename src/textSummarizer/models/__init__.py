from textSummarizer.models.abstractive import AbstractiveSummarizer, ExtractiveSummarizer
from textSummarizer.models.base import BaseSummarizer
from textSummarizer.models.registry import MODEL_REGISTRY


class ModelFactory:
    @staticmethod
    def create(
        model_name: str,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        onnx_dir: str | None = None,
    ) -> BaseSummarizer:
        key = model_name.lower().replace("_", "-")
        if key not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        spec = MODEL_REGISTRY[key]
        if spec.model_type == "extractive":
            return ExtractiveSummarizer(spec)

        if onnx_dir:
            from textSummarizer.models.onnx_summarizer import ONNXSummarizer

            return ONNXSummarizer(model_dir=onnx_dir, spec=spec)

        return AbstractiveSummarizer(
            spec=spec,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

    @staticmethod
    def list_models() -> dict[str, str]:
        return {name: spec.description for name, spec in MODEL_REGISTRY.items()}
