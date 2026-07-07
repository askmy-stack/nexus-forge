"""G-Eval style LLM-as-judge evaluation (optional deepeval dependency)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def geval_score(
    source: str,
    summary: str,
    criteria: str = "Summarization quality on coherence, faithfulness, fluency, and relevance.",
) -> dict[str, float | str]:
    """Score a summary using G-Eval when deepeval is installed, else heuristic fallback."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase

        metric = GEval(
            name="Summarization",
            criteria=criteria,
            evaluation_params=["input", "actual_output"],
        )
        test_case = LLMTestCase(input=source, actual_output=summary)
        metric.measure(test_case)
        return {
            "geval_score": float(metric.score or 0.0),
            "geval_reason": str(metric.reason or ""),
            "method": "deepeval",
        }
    except ImportError:
        logger.debug("deepeval not installed; using heuristic G-Eval fallback")
        overlap = len(set(source.lower().split()) & set(summary.lower().split()))
        denom = max(len(set(source.lower().split())), 1)
        score = min(1.0, overlap / denom)
        return {
            "geval_score": round(score, 3),
            "geval_reason": "Heuristic token-overlap fallback (install deepeval for full G-Eval)",
            "method": "heuristic",
        }
