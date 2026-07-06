import pytest

from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.loop import SummarizationLoop
from textSummarizer.grading.rubric import GradingRubric, RubricScore


def test_rubric_score_average():
    score = RubricScore(coherence=4, faithfulness=3, fluency=5, relevance=4)
    assert score.average == 4.0


def test_rubric_score_validation():
    with pytest.raises(ValueError, match="coherence"):
        RubricScore(coherence=6, faithfulness=3, fluency=3, relevance=3)


def test_rubric_passes():
    rubric = GradingRubric(threshold=3.5)
    passing = RubricScore(coherence=4, faithfulness=4, fluency=4, relevance=4)
    failing = RubricScore(coherence=2, faithfulness=2, fluency=2, relevance=2)
    assert rubric.passes(passing)
    assert not rubric.passes(failing)


def test_llm_judge_heuristic():
    judge = LLMJudge(use_llm=False)
    source = (
        "Artificial intelligence is transforming healthcare and finance. "
        "Machine learning models detect diseases from medical images."
    )
    summary = "AI transforms healthcare. Machine learning detects diseases from images."
    score = judge.grade(source, summary)
    assert 1 <= score.coherence <= 5
    assert 1 <= score.faithfulness <= 5
    assert score.average >= 2.0


def test_summarization_loop():
    loop = SummarizationLoop(model="extractive", max_iterations=1)
    source = "Sentence one about AI. Sentence two about machine learning."
    result = loop.run(source, max_length=64)
    assert result.summary
    assert result.score.average >= 1.0
    assert len(result.history) >= 1
