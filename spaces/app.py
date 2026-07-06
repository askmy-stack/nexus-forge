import gradio as gr

from textSummarizer.models import ModelFactory

MODELS = ["extractive", "bart", "t5", "flan-t5", "pegasus"]
STRATEGIES = ["stuff", "map_reduce", "refine"]


def summarize(text: str, model: str, strategy: str, max_length: int) -> str:
    if not text.strip():
        return "Please provide text to summarize."
    summarizer = ModelFactory.create(model)
    return summarizer.summarize(text, max_length=max_length, strategy=strategy)


demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Text", lines=10, placeholder="Paste article or dialogue here..."),
        gr.Dropdown(choices=MODELS, value="extractive", label="Model"),
        gr.Dropdown(choices=STRATEGIES, value="stuff", label="Strategy"),
        gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max length"),
    ],
    outputs=gr.Textbox(label="Summary", lines=6),
    title="NLP Text Summarization",
    description="Compare extractive and abstractive summarization models.",
)

if __name__ == "__main__":
    demo.launch()
