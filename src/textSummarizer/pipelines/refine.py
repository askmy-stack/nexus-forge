from textSummarizer.models.base import BaseSummarizer
from textSummarizer.pipelines.chunking import semantic_chunk
from textSummarizer.pipelines.stuff import stuff_summarize


def refine_summarize(
    text: str,
    summarizer: BaseSummarizer,
    max_length: int = 128,
    chunk_size: int = 800,
) -> str:
    chunks = semantic_chunk(text, chunk_size=chunk_size, overlap=0)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return stuff_summarize(text, summarizer, max_length=max_length)

    running_summary = stuff_summarize(chunks[0], summarizer, max_length=max_length)
    for chunk in chunks[1:]:
        prompt = (
            f"Existing summary:\n{running_summary}\n\nNew content:\n{chunk}\n\nRefined summary:"
        )
        running_summary = stuff_summarize(prompt, summarizer, max_length=max_length)
    return running_summary
