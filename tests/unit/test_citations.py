"""Tests for citation span extraction."""

from textSummarizer.pipelines.citations import extract_citation_spans, summarize_with_citations


def test_extract_citation_spans():
    source = "AI transforms healthcare. Machine learning detects diseases."
    summary = "AI transforms healthcare."
    spans = extract_citation_spans(source, summary)
    assert spans
    assert spans[0].char_start >= 0
    assert "healthcare" in spans[0].text.lower()


def test_summarize_with_citations():
    source = "Climate change raises temperatures. Renewables are growing."
    summary = "Climate change raises temperatures."
    result = summarize_with_citations(source, summary)
    assert result["summary"] == summary
    assert isinstance(result["citations"], list)
