from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        trainer = ModelTrainer(config=model_trainer_config)
        trainer.train()
        logger.info("Model training completed")


if __name__ == "__main__":
    ModelTrainerTrainingPipeline().main()
