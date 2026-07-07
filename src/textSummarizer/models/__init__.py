from textSummarizer.models.abstractive import AbstractiveSummarizer, ExtractiveSummarizer
from textSummarizer.models.base import BaseSummarizer
from textSummarizer.models.cache import cache_key, get_model_cache
from textSummarizer.models.registry import MODEL_REGISTRY


class ModelFactory:
    @staticmethod
    def create(
        model_name: str,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        onnx_dir: str | None = None,
        use_cache: bool = True,
    ) -> BaseSummarizer:
        key = model_name.lower().replace("_", "-")
        if key not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        spec = MODEL_REGISTRY[key]
        cache = get_model_cache()
        lookup = cache_key(key, model_path or "", tokenizer_path or "", onnx_dir or "")

        if spec.model_type == "extractive":
            if use_cache:
                cached = cache.get(lookup)
                if cached is not None:
                    return cached
            instance: BaseSummarizer = ExtractiveSummarizer(spec)
            if use_cache:
                cache.put(lookup, instance)
            return instance

        if use_cache:
            cached = cache.get(lookup)
            if cached is not None:
                return cached

        if onnx_dir:
            from textSummarizer.models.onnx_summarizer import ONNXSummarizer

            instance: BaseSummarizer = ONNXSummarizer(model_dir=onnx_dir, spec=spec)
        else:
            instance = AbstractiveSummarizer(
                spec=spec,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )

        if use_cache:
            cache.put(lookup, instance)
        return instance

    @staticmethod
    def list_models() -> dict[str, str]:
        return {name: spec.description for name, spec in MODEL_REGISTRY.items()}
