from textSummarizer.models.base import BaseSummarizer


def stuff_summarize(text: str, summarizer: BaseSummarizer, max_length: int = 128) -> str:
    if hasattr(summarizer, "_generate"):
        return summarizer._generate(text, max_length)  # type: ignore[attr-defined]
    return summarizer.summarize(text, max_length=max_length, strategy="stuff")
