import sys
from unittest.mock import MagicMock, patch

import pytest

from textSummarizer.export.onnx import ONNX_SUPPORTED_MODELS, export_seq2seq_to_onnx
from textSummarizer.models.onnx_summarizer import ONNXSummarizer
from textSummarizer.models.registry import MODEL_REGISTRY


def test_onnx_supported_models_subset_of_registry():
    assert ONNX_SUPPORTED_MODELS.issubset(MODEL_REGISTRY.keys())


def _mock_optimum_modules():
    mock_ort_cls = MagicMock()
    mock_ort_module = MagicMock()
    mock_ort_module.ORTModelForSeq2SeqLM = mock_ort_cls
    optimum = MagicMock()
    optimum.onnxruntime = mock_ort_module
    return optimum, mock_ort_cls


@patch("transformers.AutoTokenizer")
def test_export_seq2seq_to_onnx(mock_tokenizer_cls, tmp_path):
    optimum, mock_ort_cls = _mock_optimum_modules()
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_ort_cls.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"optimum": optimum, "optimum.onnxruntime": optimum.onnxruntime}):
        output = export_seq2seq_to_onnx("bart", tmp_path / "bart-onnx")

    assert output == tmp_path / "bart-onnx"
    mock_tokenizer.save_pretrained.assert_called_once()
    mock_model.save_pretrained.assert_called_once()
    mock_ort_cls.from_pretrained.assert_called_once()


def test_export_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        export_seq2seq_to_onnx("not-a-model", "/tmp/out")


def test_export_extractive_not_supported():
    with pytest.raises(ValueError, match="does not support ONNX"):
        export_seq2seq_to_onnx("extractive", "/tmp/out")


@patch("transformers.AutoTokenizer")
def test_onnx_summarizer_generate(mock_tokenizer_cls, tmp_path):
    optimum, mock_ort_cls = _mock_optimum_modules()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "ONNX summary."
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock_ort_cls.from_pretrained.return_value = mock_model

    spec = MODEL_REGISTRY["bart"]
    with patch.dict(sys.modules, {"optimum": optimum, "optimum.onnxruntime": optimum.onnxruntime}):
        summarizer = ONNXSummarizer(model_dir=tmp_path, spec=spec)
        summary = summarizer.summarize("Long article about AI.", max_length=32)

    assert summary == "ONNX summary."
    mock_model.generate.assert_called_once()
