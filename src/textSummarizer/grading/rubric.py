from dataclasses import dataclass, field


@dataclass
class RubricScore:
    coherence: int
    faithfulness: int
    fluency: int
    relevance: int
    feedback: str = ""

    DIMENSIONS = ("coherence", "faithfulness", "fluency", "relevance")

    def __post_init__(self):
        for dim in self.DIMENSIONS:
            value = getattr(self, dim)
            if not 1 <= value <= 5:
                raise ValueError(f"{dim} must be between 1 and 5, got {value}")

    @property
    def average(self) -> float:
        return sum(getattr(self, dim) for dim in self.DIMENSIONS) / len(self.DIMENSIONS)

    def to_dict(self) -> dict:
        return {
            "coherence": self.coherence,
            "faithfulness": self.faithfulness,
            "fluency": self.fluency,
            "relevance": self.relevance,
            "average": round(self.average, 2),
            "feedback": self.feedback,
        }


@dataclass
class GradingRubric:
    """Subjective grading rubric for summarization quality."""

    dimensions: tuple[str, ...] = RubricScore.DIMENSIONS
    threshold: float = 3.5
    descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "coherence": "Logical flow and internal consistency of the summary",
            "faithfulness": "Factual alignment with the source material",
            "fluency": "Grammar, readability, and natural phrasing",
            "relevance": "Coverage of the most important source content",
        }
    )

    def passes(self, score: RubricScore) -> bool:
        return score.average >= self.threshold
