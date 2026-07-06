from textSummarizer.models import ModelFactory
from textSummarizer.pipelines.chunking import semantic_chunk
from textSummarizer.pipelines.map_reduce import map_reduce_summarize


def test_semantic_chunk_splits_long_text():
    text = "Paragraph one.\n\n" + "Word " * 2000
    chunks = semantic_chunk(text, chunk_size=200, overlap=20)
    assert len(chunks) >= 2


def test_semantic_chunk_splits_oversized_paragraph():
    text = "Word " * 500
    chunks = semantic_chunk(text, chunk_size=100, overlap=10)
    assert len(chunks) >= 4
    assert all(len(chunk.split()) <= 110 for chunk in chunks)


def test_map_reduce_with_extractive():
    summarizer = ModelFactory.create("extractive")
    text = ". ".join([f"Sentence number {i} with content" for i in range(30)])
    summary = map_reduce_summarize(text, summarizer, max_length=64, chunk_size=50)
    assert summary
