"""Citation span extraction for grounded summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CitationSpan:
    sentence_index: int
    char_start: int
    char_end: int
    text: str

    def to_dict(self) -> dict:
        return {
            "sentence_index": self.sentence_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
        }


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    sentences: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?]+[.!?]?", text):
        start, end = match.span()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append((start, end, sentence))
    return sentences


def extract_citation_spans(
    source: str, summary: str, min_overlap: float = 0.3
) -> list[CitationSpan]:
    """Map summary sentences to supporting source spans via token overlap."""
    source_sentences = _split_sentences(source)
    summary_sentences = _split_sentences(summary)
    citations: list[CitationSpan] = []

    for _, (_, _, summary_sentence) in enumerate(summary_sentences):
        summary_tokens = set(re.findall(r"\w+", summary_sentence.lower()))
        if not summary_tokens:
            continue

        best: tuple[float, int, int, int, str] | None = None
        for src_idx, (char_start, char_end, src_sentence) in enumerate(source_sentences):
            src_tokens = set(re.findall(r"\w+", src_sentence.lower()))
            if not src_tokens:
                continue
            overlap = len(summary_tokens & src_tokens) / len(summary_tokens)
            if overlap >= min_overlap and (best is None or overlap > best[0]):
                best = (overlap, src_idx, char_start, char_end, src_sentence)

        if best is not None:
            _, src_idx, char_start, char_end, src_sentence = best
            citations.append(
                CitationSpan(
                    sentence_index=src_idx,
                    char_start=char_start,
                    char_end=char_end,
                    text=src_sentence,
                )
            )

    return citations


def summarize_with_citations(
    source: str,
    summary: str,
) -> dict:
    """Return summary plus citation spans for grounding."""
    spans = extract_citation_spans(source, summary)
    return {
        "summary": summary,
        "citations": [span.to_dict() for span in spans],
    }
