from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.models import ModelFactory


class PredictionPipeline:
    def __init__(self, model_name: str = "pegasus"):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model_name = model_name

    def predict(self, text: str, strategy: str = "stuff", max_length: int = 128) -> str:
        if self.model_name == "pegasus" and self.config.model_path.exists():
            summarizer = ModelFactory.create(
                "pegasus",
                model_path=str(self.config.model_path),
                tokenizer_path=str(self.config.tokenizer_path),
            )
        else:
            summarizer = ModelFactory.create(self.model_name)

        return summarizer.summarize(text, max_length=max_length, strategy=strategy)
