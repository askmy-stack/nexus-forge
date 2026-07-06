from dataclasses import dataclass, field

from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.rubric import GradingRubric, RubricScore
from textSummarizer.models import ModelFactory


@dataclass
class LoopResult:
    summary: str
    score: RubricScore
    iterations: int
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "score": self.score.to_dict(),
            "iterations": self.iterations,
            "history": self.history,
        }


class SummarizationLoop:
    """Iteratively summarize, grade, and refine until quality threshold is met."""

    def __init__(
        self,
        model: str = "extractive",
        rubric: GradingRubric | None = None,
        judge: LLMJudge | None = None,
        max_iterations: int = 2,
    ):
        self.model = model
        self.rubric = rubric or GradingRubric()
        self.judge = judge or LLMJudge(use_llm=False)
        self.max_iterations = max_iterations
        self._summarizer = None

    def _get_summarizer(self):
        if self._summarizer is None:
            self._summarizer = ModelFactory.create(self.model)
        return self._summarizer

    def run(
        self,
        source: str,
        max_length: int = 128,
        strategy: str = "stuff",
    ) -> LoopResult:
        summarizer = self._get_summarizer()
        history: list[dict] = []
        summary = summarizer.summarize(source, max_length=max_length, strategy=strategy)
        score = self.judge.grade(source, summary)
        history.append({"iteration": 0, "summary": summary, "score": score.to_dict()})

        iteration = 0
        while not self.rubric.passes(score) and iteration < self.max_iterations:
            iteration += 1
            refined_length = min(max_length + 32 * iteration, 256)
            refined_strategy = "map_reduce" if strategy == "stuff" else strategy
            summary = summarizer.summarize(
                source,
                max_length=refined_length,
                strategy=refined_strategy,
            )
            score = self.judge.grade(source, summary)
            history.append({"iteration": iteration, "summary": summary, "score": score.to_dict()})

        return LoopResult(
            summary=summary,
            score=score,
            iterations=iteration,
            history=history,
        )
