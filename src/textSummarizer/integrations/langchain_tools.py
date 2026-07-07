"""LangChain tool adapters for SummarizeHub."""

from __future__ import annotations

import json
from typing import Any

from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.models import ModelFactory


def _tool_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def get_summarizehub_tools() -> list:
    """Return LangChain StructuredTool instances for SummarizeHub."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install langchain-core"
        ) from exc

    def summarize_text(text: str, model: str = "extractive", strategy: str = "stuff") -> str:
        summarizer = ModelFactory.create(model)
        summary = summarizer.summarize(text, strategy=strategy)
        return _tool_result({"summary": summary, "model": model, "strategy": strategy})

    def grade_summary(source: str, summary: str, threshold: float = 3.5) -> str:
        rubric = GradingRubric(threshold=threshold)
        judge = LLMJudge(use_llm=False)
        score = judge.grade(source, summary)
        return _tool_result(
            {
                "score": score.to_dict(),
                "passes": rubric.passes(score),
                "threshold": threshold,
            }
        )

    def list_models() -> str:
        return _tool_result(ModelFactory.list_models())

    return [
        StructuredTool.from_function(
            func=summarize_text,
            name="summarize_text",
            description="Summarize plain text using extractive or abstractive models.",
        ),
        StructuredTool.from_function(
            func=grade_summary,
            name="grade_summary",
            description="Grade a summary against its source using a subjective rubric.",
        ),
        StructuredTool.from_function(
            func=list_models,
            name="list_models",
            description="List available summarization models.",
        ),
    ]
