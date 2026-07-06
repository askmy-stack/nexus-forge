import os
from pathlib import Path

from textSummarizer.entity import DataValidationConfig
from textSummarizer.logging import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        dataset_root = Path("artifacts") / "data_ingestion" / "samsum_dataset"
        if not dataset_root.exists():
            logger.error(f"Dataset root not found: {dataset_root}")
            self._write_status(False)
            return False

        all_files = os.listdir(dataset_root)
        missing = [f for f in self.config.ALL_REQUIRED_FILES if f not in all_files]
        validation_status = len(missing) == 0

        if missing:
            logger.error(f"Missing required dataset splits: {missing}")
        else:
            logger.info("All required dataset splits present")

        self._write_status(validation_status)
        return validation_status

    def _write_status(self, status: bool) -> None:
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.STATUS_FILE, "w") as f:
            f.write(f"Validation status: {status}")
