from unittest.mock import patch

from textSummarizer.models import ModelFactory
from textSummarizer.pipelines.hierarchical import hierarchical_summarize
from textSummarizer.pipelines.rag import rag_summarize, retrieve_chunks


def test_hierarchical_with_extractive():
    summarizer = ModelFactory.create("extractive")
    text = ". ".join([f"Sentence number {i} with useful content here" for i in range(30)])
    summary = hierarchical_summarize(text, summarizer, max_length=64, chunk_size=50)
    assert summary


def test_hierarchical_single_chunk_passthrough():
    summarizer = ModelFactory.create("extractive")
    text = "Short text only."
    summary = hierarchical_summarize(text, summarizer, max_length=64)
    assert summary == "Short text only."


@patch("textSummarizer.pipelines.rag.retrieve_chunks")
def test_rag_with_mocked_retrieval(mock_retrieve):
    mock_retrieve.return_value = ["Chunk about AI.", "Chunk about healthcare."]
    summarizer = ModelFactory.create("extractive")
    text = ". ".join([f"Topic {i} discusses technology and science" for i in range(20)])
    summary = rag_summarize(text, summarizer, max_length=64, chunk_size=30)
    assert summary
    mock_retrieve.assert_called_once()


@patch("textSummarizer.pipelines.rag._embedding_scores")
@patch("textSummarizer.pipelines.rag._bm25_scores")
def test_retrieve_chunks_ranks_by_combined_score(mock_bm25, mock_embed):
    mock_bm25.return_value = [0.1, 0.9, 0.5]
    mock_embed.return_value = [0.9, 0.1, 0.5]
    chunks = ["low", "high bm25", "mixed"]
    selected = retrieve_chunks("query text", chunks, top_k=2, use_embeddings=True)
    assert len(selected) == 2
    assert "high bm25" in selected
