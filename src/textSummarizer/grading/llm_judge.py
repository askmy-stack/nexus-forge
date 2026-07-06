import logging
import re

from textSummarizer.grading.rubric import RubricScore

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "if",
    "so",
    "than",
    "then",
    "there",
    "their",
    "they",
    "we",
    "you",
    "he",
    "she",
}


def _tokenize(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"\w+", text)}
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _clamp(score: float) -> int:
    return max(1, min(5, round(score)))


class LLMJudge:
    """Score summaries using heuristics with optional lightweight LLM refinement."""

    def __init__(self, use_llm: bool = False, llm_model: str = "google/flan-t5-small"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._generator = None

    def _load_generator(self):
        if self._generator is not None:
            return self._generator
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("LLM judge requires transformers") from exc
        logger.info("Loading LLM judge model: %s", self.llm_model)
        self._generator = pipeline("text2text-generation", model=self.llm_model)
        return self._generator

    def _heuristic_score(self, source: str, summary: str) -> RubricScore:
        source_tokens = _tokenize(source)
        summary_tokens = _tokenize(summary)
        overlap = len(source_tokens & summary_tokens) / max(len(summary_tokens), 1)

        sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]
        coherence = 3.0
        if len(sentences) >= 2:
            coherence += 0.5
        if len(sentences) <= 1 and len(summary.split()) > 40:
            coherence -= 1.0

        faithfulness = 2.0 + overlap * 3.0
        if len(summary_tokens - source_tokens) > len(summary_tokens) * 0.6:
            faithfulness -= 1.0

        words = summary.split()
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        fluency = 3.0
        if 4 <= avg_len <= 8:
            fluency += 0.5
        if re.search(r"\b(\w+)\s+\1\b", summary.lower()):
            fluency -= 0.5

        relevance = 2.0 + min(overlap * 4.0, 3.0)
        if len(summary.split()) < 5:
            relevance -= 1.0

        feedback_parts = []
        if faithfulness < 3:
            feedback_parts.append("Summary may contain content not grounded in the source.")
        if relevance < 3:
            feedback_parts.append("Summary may miss key source points.")
        if coherence < 3:
            feedback_parts.append("Summary structure could be clearer.")
        if fluency < 3:
            feedback_parts.append("Summary phrasing could be more fluent.")

        return RubricScore(
            coherence=_clamp(coherence),
            faithfulness=_clamp(faithfulness),
            fluency=_clamp(fluency),
            relevance=_clamp(relevance),
            feedback=(
                " ".join(feedback_parts) if feedback_parts else "Summary meets baseline quality."
            ),
        )

    def _llm_refine_feedback(self, source: str, summary: str, score: RubricScore) -> str:
        try:
            generator = self._load_generator()
            prompt = (
                f"summarize evaluation: source='{source[:300]}' "
                f"summary='{summary[:200]}' "
                f"scores coherence={score.coherence} faithfulness={score.faithfulness} "
                f"fluency={score.fluency} relevance={score.relevance}. "
                "Give one improvement tip:"
            )
            result = generator(prompt, max_new_tokens=40, do_sample=False)
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "")
                if text.strip():
                    return text.strip()
        except Exception as exc:
            logger.debug("LLM refinement unavailable: %s", exc)
        return score.feedback

    def grade(self, source: str, summary: str) -> RubricScore:
        score = self._heuristic_score(source, summary)
        if self.use_llm:
            score.feedback = self._llm_refine_feedback(source, summary, score)
        return score
