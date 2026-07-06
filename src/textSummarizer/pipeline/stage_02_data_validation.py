from textSummarizer.components.data_validation import DataValidation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class DataValidationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        if not data_validation.validate_all_files_exist():
            raise RuntimeError("Data validation failed")
        logger.info("Data validation completed")


if __name__ == "__main__":
    DataValidationTrainingPipeline().main()
