"""Tests for multi-document RAG."""

from unittest.mock import MagicMock, patch

from textSummarizer.pipelines.rag import rag_summarize_documents


@patch("textSummarizer.pipelines.rag.retrieve_chunks")
@patch("textSummarizer.pipelines.rag.stuff_summarize")
def test_rag_summarize_documents(mock_stuff, mock_retrieve):
    mock_retrieve.return_value = ["[doc-0] AI in healthcare", "[doc-1] Climate policy"]
    mock_stuff.return_value = "Combined summary."

    summarizer = MagicMock()
    result = rag_summarize_documents(
        ["AI transforms healthcare. " * 50, "Climate policy is evolving. " * 50],
        summarizer,
        max_length=64,
    )
    assert result == "Combined summary."
    mock_retrieve.assert_called_once()
