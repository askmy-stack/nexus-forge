"""ONNX Runtime inference wrapper for exported seq2seq models."""

from pathlib import Path

from textSummarizer.models.base import BaseSummarizer
from textSummarizer.models.registry import ModelSpec
from textSummarizer.pipelines.hierarchical import hierarchical_summarize
from textSummarizer.pipelines.map_reduce import map_reduce_summarize
from textSummarizer.pipelines.rag import rag_summarize
from textSummarizer.pipelines.refine import refine_summarize
from textSummarizer.pipelines.stuff import stuff_summarize


class ONNXSummarizer(BaseSummarizer):
    """Summarize text with an ONNX-exported seq2seq model."""

    def __init__(self, model_dir: str | Path, spec: ModelSpec):
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        self.spec = spec
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(self.model_dir)

    def _generate(self, text: str, max_length: int) -> str:
        input_text = text
        if self.spec.requires_prefix and not text.startswith(self.spec.requires_prefix):
            input_text = f"{self.spec.requires_prefix}{text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.spec.max_input_tokens,
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str, max_length: int = 128, strategy: str = "stuff") -> str:
        if strategy == "map_reduce":
            return map_reduce_summarize(text, self, max_length=max_length)
        if strategy == "refine":
            return refine_summarize(text, self, max_length=max_length)
        if strategy == "hierarchical":
            return hierarchical_summarize(text, self, max_length=max_length)
        if strategy == "rag":
            return rag_summarize(text, self, max_length=max_length)
        return stuff_summarize(text, self, max_length=max_length)
