"""RAPTOR-style hierarchical summarization: chunk → summarize → summarize summaries."""

from textSummarizer.models.base import BaseSummarizer
from textSummarizer.pipelines.chunking import semantic_chunk
from textSummarizer.pipelines.stuff import stuff_summarize


def hierarchical_summarize(
    text: str,
    summarizer: BaseSummarizer,
    max_length: int = 128,
    chunk_size: int = 800,
    overlap: int = 100,
    group_size: int = 3,
    max_levels: int = 4,
) -> str:
    """Recursively summarize chunks until a single summary remains."""
    chunks = semantic_chunk(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) <= 1:
        return stuff_summarize(text, summarizer, max_length=max_length)

    summaries = [stuff_summarize(chunk, summarizer, max_length=max_length) for chunk in chunks]

    level = 0
    while len(summaries) > 1 and level < max_levels:
        next_level: list[str] = []
        for index in range(0, len(summaries), group_size):
            group = "\n".join(summaries[index : index + group_size])
            next_level.append(stuff_summarize(group, summarizer, max_length=max_length))
        summaries = next_level
        level += 1

    return stuff_summarize("\n".join(summaries), summarizer, max_length=max_length)
