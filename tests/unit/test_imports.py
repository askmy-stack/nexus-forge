from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


def test_constants_exist():
    assert CONFIG_FILE_PATH.exists()
    assert PARAMS_FILE_PATH.exists()


def test_configuration_manager_loads():
    config = ConfigurationManager()
    ingestion = config.get_data_ingestion_config()
    assert "summarizer-data" in ingestion.source_URL or ingestion.source_URL.endswith(".zip")
