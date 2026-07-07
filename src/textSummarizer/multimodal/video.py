import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from textSummarizer.models import ModelFactory
from textSummarizer.multimodal.audio import AudioSummarizer
from textSummarizer.multimodal.image import ImageSummarizer
from textSummarizer.pipelines.map_reduce import map_reduce_summarize

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAMES = 20
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not available on PATH."""


class VideoSummarizer:
    """Extract audio and keyframes from video, merge ASR + captions, then summarize."""

    def __init__(
        self,
        text_model: str = "extractive",
        whisper_model: str | None = None,
        caption_model: str | None = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ):
        self.text_model = text_model
        self.max_frames = max_frames
        self._audio = AudioSummarizer(text_model=text_model)
        self._image = ImageSummarizer(text_model=text_model)
        if whisper_model:
            self._audio.whisper_model = whisper_model
        if caption_model:
            self._image.caption_model = caption_model

    @staticmethod
    def _ensure_ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FFmpegNotFoundError(
                "ffmpeg is required for video summarization but was not found on PATH. "
                "Install ffmpeg: https://ffmpeg.org/download.html"
            )
        return ffmpeg

    def _resolve_video_path(self, path: str | None, data: bytes | None) -> tuple[str, Path | None]:
        if path is not None:
            return path, None
        if data is not None:
            suffix = ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            return str(tmp_path), tmp_path
        raise ValueError("Video input requires path or bytes")

    def _run_ffmpeg(self, args: list[str]) -> None:
        ffmpeg = self._ensure_ffmpeg()
        completed = subprocess.run(
            [ffmpeg, *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"ffmpeg failed: {stderr or 'unknown error'}")

    def _extract_audio(self, video_path: str, output_wav: Path) -> None:
        self._run_ffmpeg(
            [
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_wav),
            ]
        )

    def _extract_keyframes(self, video_path: str, output_dir: Path) -> list[tuple[float, Path]]:
        """Extract scene-change keyframes with timestamps via ffmpeg scene filter."""
        pattern = output_dir / "frame_%04d.jpg"
        ffmpeg = self._ensure_ffmpeg()
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                video_path,
                "-vf",
                "select='gt(scene,0.3)',showinfo",
                "-vsync",
                "vfr",
                "-frames:v",
                str(self.max_frames),
                str(pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"ffmpeg failed: {stderr or 'unknown error'}")

        timestamps: list[float] = []
        for line in completed.stderr.splitlines():
            match = re.search(r"pts_time:([\d.]+)", line)
            if match:
                timestamps.append(float(match.group(1)))

        frames = sorted(output_dir.glob("frame_*.jpg"))
        return [
            (timestamps[index] if index < len(timestamps) else float(index), frame_path)
            for index, frame_path in enumerate(frames)
        ]

    @staticmethod
    def _build_chapters(
        segments: list[dict[str, float | str]],
        frame_captions: list[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        """Build chapter markers from speech segments and visual scene changes."""
        chapters: list[dict[str, float | str]] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if text:
                chapters.append(
                    {
                        "timestamp": float(segment["start"]),
                        "type": "speech",
                        "title": text[:80],
                    }
                )
        for frame in frame_captions:
            caption = str(frame.get("caption", "")).strip()
            if caption:
                chapters.append(
                    {
                        "timestamp": float(frame["timestamp"]),
                        "type": "visual",
                        "title": caption[:80],
                    }
                )
        chapters.sort(key=lambda item: float(item["timestamp"]))
        return chapters

    @staticmethod
    def _merge_content(
        segments: list[dict[str, float | str]],
        frame_captions: list[dict[str, float | str]],
    ) -> str:
        events: list[tuple[float, str, str]] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if text:
                events.append((float(segment["start"]), "Speech", text))
        for frame in frame_captions:
            caption = str(frame.get("caption", "")).strip()
            if caption:
                events.append((float(frame["timestamp"]), "Visual", caption))

        events.sort(key=lambda item: item[0])
        lines = ["# Video Content Analysis", ""]
        for timestamp, label, text in events:
            lines.append(f"[{timestamp:05.1f}s] {label}: {text}")
        return "\n".join(lines).strip()

    def build_document(self, path: str | None = None, data: bytes | None = None) -> dict:
        video_path, video_tmp = self._resolve_video_path(path, data)
        work_dir = Path(tempfile.mkdtemp(prefix="summarizehub-video-"))
        audio_wav = work_dir / "audio.wav"
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._extract_audio(video_path, audio_wav)
            segments = self._audio.transcribe_segments(path=str(audio_wav))
            frames = self._extract_keyframes(video_path, frames_dir)
            frame_captions: list[dict[str, float | str]] = []
            for timestamp, frame_path in frames:
                caption = self._image.caption(path=str(frame_path))
                frame_captions.append({"timestamp": timestamp, "caption": caption})

            document = self._merge_content(segments, frame_captions)
            chapters = self._build_chapters(segments, frame_captions)
            transcript = " ".join(
                str(segment.get("text", "")).strip() for segment in segments if segment.get("text")
            ).strip()
            visual_captions = "; ".join(
                str(frame.get("caption", "")).strip()
                for frame in frame_captions
                if frame.get("caption")
            )
            return {
                "document": document,
                "transcript": transcript,
                "visual_captions": visual_captions,
                "segments": segments,
                "frame_captions": frame_captions,
                "chapters": chapters,
            }
        finally:
            if video_tmp is not None:
                video_tmp.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)

    def summarize(
        self,
        path: str | None = None,
        data: bytes | None = None,
        max_length: int = 128,
        strategy: str = "stuff",
    ) -> dict[str, str | list]:
        content = self.build_document(path=path, data=data)
        document = str(content["document"])
        summarizer = ModelFactory.create(self.text_model)
        if strategy == "map_reduce":
            summary = map_reduce_summarize(document, summarizer, max_length=max_length)
        else:
            summary = summarizer.summarize(document, max_length=max_length, strategy=strategy)
        return {
            "document": document,
            "transcript": str(content["transcript"]),
            "visual_captions": str(content["visual_captions"]),
            "summary": summary,
            "model": self.text_model,
            "segments": content["segments"],
            "frame_captions": content["frame_captions"],
            "chapters": content.get("chapters", []),
        }
