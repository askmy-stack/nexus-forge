from unittest.mock import patch

import pytest

from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.router import MultimodalRouter
from textSummarizer.multimodal.video import FFmpegNotFoundError, VideoSummarizer


def test_video_build_chapters():
    segments = [{"start": 1.0, "end": 3.0, "text": "Introduction to the topic."}]
    frame_captions = [{"timestamp": 0.5, "caption": "Title slide."}]
    chapters = VideoSummarizer._build_chapters(segments, frame_captions)
    assert len(chapters) == 2
    assert chapters[0]["type"] == "visual"
    assert chapters[1]["type"] == "speech"


def test_merge_content_orders_events_by_timestamp():
    segments = [
        {"start": 5.0, "end": 7.0, "text": "Later speech."},
        {"start": 1.0, "end": 2.0, "text": "Early speech."},
    ]
    frame_captions = [
        {"timestamp": 3.0, "caption": "A person on stage."},
        {"timestamp": 0.0, "caption": "Opening title card."},
    ]

    document = VideoSummarizer._merge_content(segments, frame_captions)

    assert "[000.0s] Visual: Opening title card." in document
    assert document.index("Early speech.") < document.index("A person on stage.")
    assert document.index("A person on stage.") < document.index("Later speech.")


@patch("textSummarizer.multimodal.video.VideoSummarizer._extract_keyframes")
@patch("textSummarizer.multimodal.video.VideoSummarizer._extract_audio")
@patch("textSummarizer.multimodal.video.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("textSummarizer.multimodal.audio.AudioSummarizer.transcribe_segments")
@patch("textSummarizer.multimodal.image.ImageSummarizer.caption")
def test_video_summarizer_mock(
    mock_caption,
    mock_transcribe_segments,
    mock_which,
    mock_extract_audio,
    mock_extract_keyframes,
    tmp_path,
):
    mock_transcribe_segments.return_value = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to the demo."}
    ]
    frame_one = tmp_path / "frame_0000.jpg"
    frame_two = tmp_path / "frame_0001.jpg"
    frame_one.write_bytes(b"frame-1")
    frame_two.write_bytes(b"frame-2")
    mock_extract_keyframes.return_value = [(0.0, frame_one), (1.0, frame_two)]
    mock_caption.side_effect = ["A presenter at a podium.", "A slide with charts."]

    summarizer = VideoSummarizer(text_model="extractive", max_frames=2)
    result = summarizer.summarize(path="/tmp/fake-video.mp4", strategy="stuff")

    assert "Welcome to the demo." in result["document"]
    assert "A presenter at a podium." in result["document"]
    assert result["transcript"] == "Welcome to the demo."
    assert "presenter" in result["visual_captions"]
    assert result["summary"]
    assert "chapters" in result
    mock_extract_audio.assert_called_once()


@patch("textSummarizer.multimodal.video.shutil.which", return_value=None)
def test_video_summarizer_missing_ffmpeg(mock_which):
    summarizer = VideoSummarizer(text_model="extractive")
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg is required"):
        summarizer.build_document(path="/tmp/fake-video.mp4")


@patch("textSummarizer.multimodal.router.VideoSummarizer.summarize")
def test_router_video(mock_summarize):
    mock_summarize.return_value = {
        "document": "[0000.0s] Speech: Hello.",
        "transcript": "Hello.",
        "visual_captions": "A scene.",
        "summary": "Video summary.",
        "model": "extractive",
        "segments": [],
        "frame_captions": [],
        "chapters": [{"timestamp": 0.0, "type": "speech", "title": "Hello."}],
    }
    router = MultimodalRouter(text_model="extractive")
    result = router.summarize(
        MultimodalInput(input_type=InputType.VIDEO, path="/tmp/video.mp4"),
        strategy="map_reduce",
    )

    assert result["input_type"] == "video"
    assert result["summary"] == "Video summary."
    assert result["document"]
    mock_summarize.assert_called_once()
