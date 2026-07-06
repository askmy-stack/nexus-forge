from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
        logger.info("Data transformation completed")


if __name__ == "__main__":
    DataTransformationTrainingPipeline().main()
