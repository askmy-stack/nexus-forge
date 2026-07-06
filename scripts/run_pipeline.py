"""Run the full 5-stage training pipeline."""

from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

STAGES = [
    ("Data Ingestion", DataIngestionTrainingPipeline),
    ("Data Validation", DataValidationTrainingPipeline),
    ("Data Transformation", DataTransformationTrainingPipeline),
    ("Model Trainer", ModelTrainerTrainingPipeline),
]


def main() -> None:
    for stage_name, pipeline_cls in STAGES:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline_cls().main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")


if __name__ == "__main__":
    main()
