from textSummarizer.evaluation.metrics import EvaluationSuite


def test_evaluation_suite_rouge():
    suite = EvaluationSuite(tier=1)
    predictions = ["the cat sat on the mat"]
    references = ["the cat is on the mat"]
    scores = suite.evaluate(predictions, references)
    assert any(k.startswith("rouge_") for k in scores)
