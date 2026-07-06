from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.models import ModelFactory


def test_constants_exist():
    assert CONFIG_FILE_PATH.exists()
    assert PARAMS_FILE_PATH.exists()


def test_configuration_manager_loads():
    config = ConfigurationManager()
    ingestion = config.get_data_ingestion_config()
    assert "summarizer-data" in ingestion.source_URL or ingestion.source_URL.endswith(".zip")


def test_model_registry_lists_models():
    models = ModelFactory.list_models()
    assert "extractive" in models
    assert "bart" in models
    assert "pegasus" in models


def test_extractive_summarizer():
    summarizer = ModelFactory.create("extractive")
    text = (
        "Artificial intelligence is transforming industries. "
        "Machine learning enables automation at scale. "
        "Natural language processing powers modern assistants."
    )
    summary = summarizer.summarize(text, max_length=64)
    assert summary
    assert len(summary.split()) <= len(text.split())
