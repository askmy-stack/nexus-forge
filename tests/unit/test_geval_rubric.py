"""Tests for G-Eval, GPU pool, and rubric loading."""

from unittest.mock import MagicMock, patch

from textSummarizer.evaluation.metrics import EvaluationSuite
from textSummarizer.grading.geval import (
    get_geval_model,
    geval_score,
    is_geval_available,
)
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.grading.rubric_loader import list_rubrics, load_rubric_from_yaml
from textSummarizer.serving.gpu_pool import GPUJobPool, get_gpu_pool


def test_geval_heuristic_fallback():
    result = geval_score("AI transforms healthcare.", "AI transforms healthcare.", use_geval=False)
    assert "geval_score" in result
    assert result["method"] == "heuristic"
    assert 0.0 <= float(result["geval_score"]) <= 1.0


def test_geval_heuristic_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEVAL_API_KEY", raising=False)
    with patch("textSummarizer.grading.geval.is_deepeval_installed", return_value=True):
        result = geval_score("AI transforms healthcare.", "AI transforms healthcare.")
    assert result["method"] == "heuristic"


def test_geval_deepeval_with_mocked_metric(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_metric = MagicMock()
    mock_metric.score = 0.92
    mock_metric.reason = "Strong summary."
    mock_geval_cls = MagicMock(return_value=mock_metric)
    mock_gpt_model = MagicMock()

    with (
        patch("textSummarizer.grading.geval.is_deepeval_installed", return_value=True),
        patch.dict(
            "sys.modules",
            {
                "deepeval": MagicMock(),
                "deepeval.metrics": MagicMock(GEval=mock_geval_cls),
                "deepeval.models": MagicMock(GPTModel=mock_gpt_model),
                "deepeval.test_case": MagicMock(
                    LLMTestCase=MagicMock,
                    SingleTurnParams=MagicMock(INPUT="input", ACTUAL_OUTPUT="actual_output"),
                ),
            },
        ),
    ):
        result = geval_score(
            "AI transforms healthcare.",
            "AI transforms healthcare.",
            use_geval=True,
        )

    assert result["method"] == "deepeval"
    assert result["geval_score"] == 0.92
    assert result["geval_reason"] == "Strong summary."
    mock_geval_cls.assert_called_once()


def test_geval_model_env(monkeypatch):
    monkeypatch.setenv("GEVAL_MODEL", "gpt-4.1-mini")
    assert get_geval_model() == "gpt-4.1-mini"


def test_is_geval_available_requires_key_and_deepeval(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEVAL_API_KEY", raising=False)
    with patch("textSummarizer.grading.geval.is_deepeval_installed", return_value=True):
        assert is_geval_available() is False

    monkeypatch.setenv("GEVAL_API_KEY", "test")
    with patch("textSummarizer.grading.geval.is_deepeval_installed", return_value=True):
        assert is_geval_available() is True


def test_evaluation_suite_tier_4():
    suite = EvaluationSuite(tier=4)
    predictions = ["AI transforms healthcare"]
    references = ["AI changes healthcare"]
    sources = ["AI is transforming healthcare systems worldwide."]
    scores = suite.evaluate(predictions, references, sources=sources)
    assert any(k.startswith("rouge_") for k in scores)
    assert "geval_score" in scores
    assert "geval_method" in scores


def test_evaluation_suite_tier_4_deepeval(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    suite = EvaluationSuite(tier=4, use_geval=True)
    with patch(
        "textSummarizer.grading.geval.geval_score",
        return_value={"geval_score": 0.88, "method": "deepeval", "geval_reason": "ok"},
    ):
        scores = suite.evaluate(
            ["summary"],
            ["ref"],
            sources=["source text"],
        )
    assert scores["geval_score"] == 0.88
    assert scores["geval_method"] == "deepeval"


def test_gpu_pool_limits_concurrency():
    pool = GPUJobPool(max_jobs=1, cache_ttl_seconds=60)
    entered = []

    def worker():
        with pool.acquire():
            entered.append(1)
            return "done"

    import threading

    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(worker())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2


def test_gpu_pool_model_cache_ttl(monkeypatch):
    pool = GPUJobPool(max_jobs=2, cache_ttl_seconds=10, max_cached_models=2)
    calls = {"count": 0}
    clock = [1000.0]
    monkeypatch.setattr(
        "textSummarizer.serving.gpu_pool.time.monotonic",
        lambda: clock[0],
    )

    def loader():
        calls["count"] += 1
        return object()

    first = pool.get_model("bart", loader)
    assert calls["count"] == 1
    clock[0] += 11
    second = pool.get_model("bart", loader)
    assert calls["count"] == 2
    assert first is not second


def test_get_gpu_pool_singleton():
    assert get_gpu_pool() is get_gpu_pool()


def test_load_rubric_from_yaml():
    rubric = load_rubric_from_yaml("default")
    assert rubric.threshold == 3.5
    assert "coherence" in rubric.descriptions


def test_grading_rubric_from_yaml():
    rubric = GradingRubric.from_yaml("strict")
    assert rubric.threshold == 4.0


def test_list_rubrics():
    names = list_rubrics()
    assert "default" in names
    assert "strict" in names
