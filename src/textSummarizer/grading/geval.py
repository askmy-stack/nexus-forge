"""G-Eval style LLM-as-judge evaluation (optional deepeval dependency)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_GEVAL_MODEL = "gpt-4o-mini"
DEFAULT_CRITERIA = "Summarization quality on coherence, faithfulness, fluency, and relevance."


def _get_api_key() -> str | None:
    return os.getenv("GEVAL_API_KEY") or os.getenv("OPENAI_API_KEY")


def get_geval_model() -> str:
    """Return the judge model name from ``GEVAL_MODEL`` (default: gpt-4o-mini)."""
    return os.getenv("GEVAL_MODEL", DEFAULT_GEVAL_MODEL)


def is_deepeval_installed() -> bool:
    try:
        import deepeval  # noqa: F401

        return True
    except ImportError:
        return False


def is_geval_available() -> bool:
    """True when deepeval is installed and an OpenAI-compatible API key is set."""
    return is_deepeval_installed() and bool(_get_api_key())


def _heuristic_score(
    source: str, summary: str, reason: str | None = None
) -> dict[str, float | str]:
    overlap = len(set(source.lower().split()) & set(summary.lower().split()))
    denom = max(len(set(source.lower().split())), 1)
    score = min(1.0, overlap / denom)
    default_reason = (
        "Heuristic token-overlap fallback. "
        "Install deepeval (`uv pip install -e '.[eval]'`) and set OPENAI_API_KEY "
        "or GEVAL_API_KEY for full G-Eval LLM judging."
    )
    return {
        "geval_score": round(score, 3),
        "geval_reason": reason or default_reason,
        "method": "heuristic",
    }


def _deepeval_score(
    source: str,
    summary: str,
    criteria: str,
    model_name: str,
    api_key: str,
) -> dict[str, float | str]:
    from deepeval.metrics import GEval
    from deepeval.models import GPTModel
    from deepeval.test_case import LLMTestCase

    try:
        from deepeval.test_case import LLMTestCaseParams as EvalParams
    except ImportError:
        from deepeval.test_case import SingleTurnParams as EvalParams

    judge = GPTModel(model=model_name, api_key=api_key, temperature=0)
    metric = GEval(
        name="Summarization",
        criteria=criteria,
        evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT],
        model=judge,
    )
    test_case = LLMTestCase(input=source, actual_output=summary)
    metric.measure(test_case)
    return {
        "geval_score": float(metric.score or 0.0),
        "geval_reason": str(metric.reason or ""),
        "method": "deepeval",
        "model": model_name,
    }


def geval_score(
    source: str,
    summary: str,
    criteria: str = DEFAULT_CRITERIA,
    use_geval: bool | None = None,
) -> dict[str, float | str]:
    """Score a summary using G-Eval when available, else heuristic fallback.

    Full G-Eval runs when ``deepeval`` is installed and ``OPENAI_API_KEY`` or
    ``GEVAL_API_KEY`` is set. Set ``use_geval=False`` to force the heuristic.
    """
    if use_geval is False:
        return _heuristic_score(source, summary)

    if not is_geval_available():
        if use_geval is True:
            missing = []
            if not is_deepeval_installed():
                missing.append("deepeval (install with `uv pip install -e '.[eval]'`)")
            if not _get_api_key():
                missing.append("OPENAI_API_KEY or GEVAL_API_KEY")
            reason = f"G-Eval unavailable: missing {', '.join(missing)}."
            return _heuristic_score(source, summary, reason=reason)
        return _heuristic_score(source, summary)

    api_key = _get_api_key()
    assert api_key is not None
    model_name = get_geval_model()
    try:
        return _deepeval_score(source, summary, criteria, model_name, api_key)
    except Exception as exc:
        logger.warning("deepeval G-Eval failed, using heuristic fallback: %s", exc)
        return _heuristic_score(
            source,
            summary,
            reason=f"Heuristic fallback after G-Eval error: {exc}",
        )
