from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluator = ModelEvaluation(config=model_evaluation_config)
        evaluator.evaluate()
        logger.info("Model evaluation completed")


if __name__ == "__main__":
    ModelEvaluationTrainingPipeline().main()
