import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.models.base import BaseSummarizer
from textSummarizer.models.registry import ModelSpec
from textSummarizer.pipelines.hierarchical import hierarchical_summarize
from textSummarizer.pipelines.map_reduce import map_reduce_summarize
from textSummarizer.pipelines.rag import rag_summarize
from textSummarizer.pipelines.refine import refine_summarize
from textSummarizer.pipelines.stuff import stuff_summarize


def _extractive_stuff(text: str, max_length: int) -> str:
    """Score sentences by word frequency and return the top-ranked subset."""
    sentences = ExtractiveSummarizer._split_sentences(text)
    if not sentences:
        return ""

    words = re.findall(r"\w+", text.lower())
    freq: dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\w+", sentence.lower())
        if not sentence_words:
            continue
        score = sum(freq.get(w, 0) for w in sentence_words) / len(sentence_words)
        scored.append((score, idx, sentence))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top_n = max(1, min(len(scored), max(1, max_length // 20)))
    selected = sorted(scored[:top_n], key=lambda x: x[1])
    return " ".join(s for _, _, s in selected)


class AbstractiveSummarizer(BaseSummarizer):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        device: str | None = None,
    ):
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = model_path or spec.model_id
        tokenizer_id = tokenizer_path or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
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


class ExtractiveSummarizer(BaseSummarizer):
    """Lightweight extractive summarizer using sentence scoring."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def summarize(self, text: str, max_length: int = 128, strategy: str = "stuff") -> str:
        if strategy == "map_reduce":
            return map_reduce_summarize(text, self, max_length=max_length)
        if strategy == "refine":
            return refine_summarize(text, self, max_length=max_length)
        if strategy == "hierarchical":
            return hierarchical_summarize(text, self, max_length=max_length)
        if strategy == "rag":
            return rag_summarize(text, self, max_length=max_length)
        return _extractive_stuff(text, max_length)
