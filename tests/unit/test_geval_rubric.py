"""Tests for G-Eval and rubric loading."""

from textSummarizer.evaluation.metrics import EvaluationSuite
from textSummarizer.grading.geval import geval_score
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.grading.rubric_loader import list_rubrics, load_rubric_from_yaml


def test_geval_heuristic_fallback():
    result = geval_score("AI transforms healthcare.", "AI transforms healthcare.")
    assert "geval_score" in result
    assert result["method"] == "heuristic"
    assert 0.0 <= float(result["geval_score"]) <= 1.0


def test_evaluation_suite_tier_4():
    suite = EvaluationSuite(tier=4)
    predictions = ["AI transforms healthcare"]
    references = ["AI changes healthcare"]
    sources = ["AI is transforming healthcare systems worldwide."]
    scores = suite.evaluate(predictions, references, sources=sources)
    assert any(k.startswith("rouge_") for k in scores)
    assert "geval_score" in scores


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
