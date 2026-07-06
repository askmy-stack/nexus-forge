"""SummarizeHub MCP server — run with: python -m textSummarizer.mcp.server"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.models import ModelFactory
from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.router import MultimodalRouter

logger = logging.getLogger(__name__)

mcp = FastMCP("summarizehub")


@mcp.tool()
def summarize_text(
    text: str,
    model: str = "extractive",
    strategy: str = "stuff",
    max_length: int = 128,
) -> str:
    """Summarize plain text using extractive or abstractive models.

    strategy: stuff | map_reduce | refine | hierarchical | rag
    """
    summarizer = ModelFactory.create(model)
    summary = summarizer.summarize(text, max_length=max_length, strategy=strategy)
    return json.dumps({"summary": summary, "model": model, "strategy": strategy})


@mcp.tool()
def summarize_image(
    path: str | None = None,
    base64_data: str | None = None,
    model: str = "extractive",
    max_length: int = 128,
) -> str:
    """Caption an image with BLIP and summarize the caption."""
    router = MultimodalRouter(text_model=model)
    result = router.summarize(
        MultimodalInput(
            input_type=InputType.IMAGE,
            path=path,
            base64_data=base64_data,
        ),
        max_length=max_length,
    )
    return json.dumps(result)


@mcp.tool()
def summarize_audio(
    path: str | None = None,
    base64_data: str | None = None,
    model: str = "extractive",
    max_length: int = 128,
) -> str:
    """Transcribe audio with Whisper and summarize the transcript."""
    router = MultimodalRouter(text_model=model)
    result = router.summarize(
        MultimodalInput(
            input_type=InputType.AUDIO,
            path=path,
            base64_data=base64_data,
        ),
        max_length=max_length,
    )
    return json.dumps(result)


@mcp.tool()
def summarize_video(
    path: str | None = None,
    base64_data: str | None = None,
    model: str = "extractive",
    max_length: int = 128,
    strategy: str = "map_reduce",
) -> str:
    """Extract audio/keyframes from video, merge ASR + BLIP captions, then summarize."""
    router = MultimodalRouter(text_model=model)
    result = router.summarize(
        MultimodalInput(
            input_type=InputType.VIDEO,
            path=path,
            base64_data=base64_data,
        ),
        max_length=max_length,
        strategy=strategy,
    )
    return json.dumps(result)


@mcp.tool()
def list_models() -> str:
    """List available summarization models and descriptions."""
    return json.dumps(ModelFactory.list_models())


@mcp.tool()
def grade_summary(source: str, summary: str, threshold: float = 3.5) -> str:
    """Subjective LLM-rubric grading of a summary against its source."""
    rubric = GradingRubric(threshold=threshold)
    judge = LLMJudge(use_llm=False)
    score = judge.grade(source, summary)
    return json.dumps(
        {
            "score": score.to_dict(),
            "passes": rubric.passes(score),
            "threshold": threshold,
        }
    )


def main():
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
