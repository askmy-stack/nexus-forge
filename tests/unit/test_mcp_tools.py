import importlib.util
import json
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")

from textSummarizer.mcp import server as mcp_server


def test_mcp_tools_registered():
    tool_manager = mcp_server.mcp._tool_manager
    tools = tool_manager.list_tools()
    names = {tool.name for tool in tools}
    expected = {
        "summarize_text",
        "summarize_image",
        "summarize_audio",
        "summarize_video",
        "list_models",
        "grade_summary",
    }
    assert expected.issubset(names)


def test_summarize_text_tool():
    result = mcp_server.summarize_text(
        text="AI is changing industries. Machine learning enables automation.",
        model="extractive",
    )
    data = json.loads(result)
    assert data["summary"]
    assert data["model"] == "extractive"


def test_summarize_text_hierarchical_strategy():
    result = mcp_server.summarize_text(
        text="AI is changing industries. " * 20,
        model="extractive",
        strategy="hierarchical",
    )
    data = json.loads(result)
    assert data["summary"]
    assert data["strategy"] == "hierarchical"


@patch("textSummarizer.pipelines.rag.retrieve_chunks")
def test_summarize_text_rag_strategy(mock_retrieve):
    mock_retrieve.return_value = ["Relevant chunk about AI."]
    text = "word " * 3000
    result = mcp_server.summarize_text(
        text=text,
        model="extractive",
        strategy="rag",
    )
    data = json.loads(result)
    assert data["summary"]
    assert data["strategy"] == "rag"
    mock_retrieve.assert_called_once()


def test_list_models_tool():
    result = mcp_server.list_models()
    models = json.loads(result)
    assert "extractive" in models


def test_grade_summary_tool():
    result = mcp_server.grade_summary(
        source="AI transforms healthcare and finance.",
        summary="AI transforms healthcare.",
    )
    data = json.loads(result)
    assert "score" in data
    assert "passes" in data
    assert 1 <= data["score"]["coherence"] <= 5


def test_grade_summary_tool_with_geval():
    mock_geval = {
        "geval_score": 0.85,
        "geval_reason": "Good summary.",
        "method": "deepeval",
        "model": "gpt-4o-mini",
    }
    with (
        patch("textSummarizer.mcp.server.is_geval_available", return_value=True),
        patch("textSummarizer.mcp.server.geval_score", return_value=mock_geval),
    ):
        result = mcp_server.grade_summary(
            source="AI transforms healthcare and finance.",
            summary="AI transforms healthcare.",
        )
    data = json.loads(result)
    assert data["geval"]["method"] == "deepeval"
    assert data["geval"]["geval_score"] == 0.85


@pytest.mark.parametrize(
    "tool_name",
    [
        "summarize_text",
        "summarize_image",
        "summarize_audio",
        "summarize_video",
        "list_models",
        "grade_summary",
    ],
)
def test_mcp_package_entrypoint(tool_name):
    assert importlib.util.find_spec("textSummarizer.mcp.server") is not None
    tool_manager = mcp_server.mcp._tool_manager
    names = {tool.name for tool in tool_manager.list_tools()}
    assert tool_name in names
