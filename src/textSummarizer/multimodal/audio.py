import logging
import tempfile
from pathlib import Path

from textSummarizer.models import ModelFactory
from textSummarizer.models.cache import cache_key, get_model_cache

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "openai/whisper-tiny"


class AudioSummarizer:
    """Transcribe audio with Whisper, then summarize the transcript."""

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        text_model: str = "extractive",
    ):
        self.whisper_model = whisper_model
        self.text_model = text_model
        self._pipeline = None

    def _load_pipeline(self):
        cache = get_model_cache()
        lookup = cache_key("whisper", self.whisper_model)
        cached = cache.get(lookup)
        if cached is not None:
            self._pipeline = cached
            return self._pipeline
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "Audio summarization requires transformers. "
                "Install multimodal extras: uv sync --extra multimodal"
            ) from exc
        logger.info("Loading whisper model: %s", self.whisper_model)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
        )
        cache.put(lookup, self._pipeline)
        return self._pipeline

    def _resolve_audio_path(self, path: str | None, data: bytes | None) -> tuple[str, Path | None]:
        if path is not None:
            return path, None
        if data is not None:
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            return str(tmp_path), tmp_path
        raise ValueError("Audio input requires path or bytes")

    def transcribe(self, path: str | None = None, data: bytes | None = None) -> str:
        segments = self.transcribe_segments(path=path, data=data)
        return " ".join(segment["text"] for segment in segments if segment["text"]).strip()

    def transcribe_segments(
        self, path: str | None = None, data: bytes | None = None
    ) -> list[dict[str, float | str]]:
        """Transcribe audio and return timestamped speech segments."""
        pipe = self._load_pipeline()
        audio_path, tmp = self._resolve_audio_path(path, data)
        try:
            result = pipe(audio_path, return_timestamps=True)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

        segments: list[dict[str, float | str]] = []
        if isinstance(result, dict):
            chunks = result.get("chunks") or []
            for chunk in chunks:
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                start = float(timestamp[0] or 0.0)
                end = float(timestamp[1] or start)
                text = str(chunk.get("text", "")).strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})
            if not segments:
                text = str(result.get("text", "")).strip()
                if text:
                    segments.append({"start": 0.0, "end": 0.0, "text": text})
            return segments

        text = str(result).strip()
        if text:
            return [{"start": 0.0, "end": 0.0, "text": text}]
        return []

    def summarize(
        self,
        path: str | None = None,
        data: bytes | None = None,
        max_length: int = 128,
        strategy: str = "stuff",
    ) -> dict[str, str]:
        transcript = self.transcribe(path=path, data=data)
        summarizer = ModelFactory.create(self.text_model)
        summary = summarizer.summarize(transcript, max_length=max_length, strategy=strategy)
        return {
            "transcript": transcript,
            "summary": summary,
            "model": self.text_model,
        }
