import logging
import tempfile
from pathlib import Path

from textSummarizer.models import ModelFactory

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
        pipe = self._load_pipeline()
        audio_path, tmp = self._resolve_audio_path(path, data)
        try:
            result = pipe(audio_path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

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
