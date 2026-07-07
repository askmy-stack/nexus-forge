"""Tests for LangChain integration."""

import pytest

from textSummarizer.integrations.langchain_tools import get_summarizehub_tools


def test_langchain_tools():
    langchain_core = pytest.importorskip("langchain_core")
    _ = langchain_core
    tools = get_summarizehub_tools()
    names = {tool.name for tool in tools}
    assert "summarize_text" in names
    assert "grade_summary" in names
    assert "list_models" in names
