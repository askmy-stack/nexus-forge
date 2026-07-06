"""SummarizeHub HuggingFace Space — GPU-backed abstractive summarization."""

import gradio as gr

try:
    import spaces
except ImportError:
    spaces = None  # type: ignore[assignment]

from textSummarizer.models import ModelFactory
from textSummarizer.pipelines import STRATEGIES

MODELS = ["bart", "flan-t5", "t5", "pegasus", "extractive"]
DEFAULT_MODEL = "bart"


def _summarize_impl(text: str, model: str, strategy: str, max_length: int) -> str:
    if not text.strip():
        return "Please provide text to summarize."
    summarizer = ModelFactory.create(model)
    return summarizer.summarize(text, max_length=max_length, strategy=strategy)


if spaces is not None:

    @spaces.GPU(duration=120)
    def summarize(text: str, model: str, strategy: str, max_length: int) -> str:
        return _summarize_impl(text, model, strategy, max_length)

else:

    def summarize(text: str, model: str, strategy: str, max_length: int) -> str:
        return _summarize_impl(text, model, strategy, max_length)


demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Text", lines=10, placeholder="Paste article or dialogue here..."),
        gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL, label="Model"),
        gr.Dropdown(choices=list(STRATEGIES), value="stuff", label="Strategy"),
        gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max length"),
    ],
    outputs=gr.Textbox(label="Summary", lines=6),
    title="SummarizeHub",
    description=(
        "GPU-backed abstractive summarization with BART, FLAN-T5, and long-document "
        "strategies including hierarchical and RAG."
    ),
    examples=[
        [
            (
                "Artificial intelligence is reshaping healthcare, finance, and transportation. "
                "Machine learning models can detect diseases from medical images. "
                "Natural language processing powers chatbots and document summarization."
            ),
            "bart",
            "stuff",
            128,
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()
