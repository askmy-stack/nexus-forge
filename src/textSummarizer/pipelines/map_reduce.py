from textSummarizer.models.base import BaseSummarizer
from textSummarizer.pipelines.chunking import semantic_chunk
from textSummarizer.pipelines.stuff import stuff_summarize


def map_reduce_summarize(
    text: str,
    summarizer: BaseSummarizer,
    max_length: int = 128,
    chunk_size: int = 800,
    overlap: int = 100,
) -> str:
    chunks = semantic_chunk(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) == 1:
        return stuff_summarize(text, summarizer, max_length=max_length)

    chunk_summaries = [
        stuff_summarize(chunk, summarizer, max_length=max_length) for chunk in chunks
    ]
    combined = "\n".join(chunk_summaries)
    return stuff_summarize(combined, summarizer, max_length=max_length)
