import base64
from unittest.mock import MagicMock, patch

import pytest

from textSummarizer.multimodal.audio import AudioSummarizer
from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.image import ImageSummarizer
from textSummarizer.multimodal.router import MultimodalRouter


def test_multimodal_input_base64_decode():
    raw = b"hello world"
    encoded = base64.b64encode(raw).decode()
    inp = MultimodalInput(input_type=InputType.TEXT, base64_data=encoded)
    assert inp.resolve_bytes() == raw
    assert inp.resolve_text() == "hello world"


def test_multimodal_input_data_uri():
    raw = b"image bytes"
    encoded = base64.b64encode(raw).decode()
    inp = MultimodalInput(
        input_type=InputType.IMAGE,
        base64_data=f"data:image/png;base64,{encoded}",
    )
    assert inp.resolve_bytes() == raw


@patch("textSummarizer.multimodal.image.ImageSummarizer._load_pipeline")
@patch("textSummarizer.multimodal.image.ImageSummarizer._load_image")
def test_image_summarizer_mock(mock_load_image, mock_load_pipeline):
    mock_pipe = MagicMock(return_value=[{"generated_text": "A cat on a mat."}])
    mock_load_pipeline.return_value = mock_pipe
    mock_load_image.return_value = MagicMock()

    summarizer = ImageSummarizer(text_model="extractive")
    result = summarizer.summarize(data=b"fake-image")

    assert result["caption"] == "A cat on a mat."
    assert result["summary"]
    mock_pipe.assert_called_once()


@patch("textSummarizer.multimodal.audio.AudioSummarizer._load_pipeline")
@patch("textSummarizer.multimodal.audio.AudioSummarizer._resolve_audio_path")
def test_audio_summarizer_mock(mock_resolve_path, mock_load_pipeline):
    mock_pipe = MagicMock(return_value={"text": "Hello from audio."})
    mock_load_pipeline.return_value = mock_pipe
    mock_resolve_path.return_value = ("/tmp/fake.wav", None)

    summarizer = AudioSummarizer(text_model="extractive")
    result = summarizer.summarize(data=b"fake-audio")

    assert result["transcript"] == "Hello from audio."
    assert result["summary"]
    mock_pipe.assert_called_once_with("/tmp/fake.wav")


def test_router_text():
    router = MultimodalRouter(text_model="extractive")
    result = router.summarize(
        MultimodalInput(
            input_type=InputType.TEXT,
            text="AI is changing the world. ML powers assistants.",
        )
    )
    assert result["input_type"] == "text"
    assert result["summary"]


@patch("textSummarizer.multimodal.router.ImageSummarizer.summarize")
def test_router_image(mock_summarize):
    mock_summarize.return_value = {
        "caption": "A diagram.",
        "summary": "Diagram summary.",
        "model": "extractive",
    }
    router = MultimodalRouter()
    result = router.summarize(
        MultimodalInput(input_type=InputType.IMAGE, base64_data=base64.b64encode(b"img").decode())
    )
    assert result["caption"] == "A diagram."
    assert result["summary"] == "Diagram summary."


def test_router_video_not_implemented():
    router = MultimodalRouter()
    with pytest.raises(NotImplementedError):
        router.summarize(MultimodalInput(input_type=InputType.VIDEO, path="/tmp/video.mp4"))
