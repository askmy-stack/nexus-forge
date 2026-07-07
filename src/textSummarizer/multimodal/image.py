import logging
from io import BytesIO

from textSummarizer.models import ModelFactory
from textSummarizer.models.cache import cache_key, get_model_cache

logger = logging.getLogger(__name__)

DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"


class ImageSummarizer:
    """Caption an image with BLIP, then summarize the caption with a text model."""

    def __init__(
        self,
        caption_model: str = DEFAULT_BLIP_MODEL,
        text_model: str = "extractive",
    ):
        self.caption_model = caption_model
        self.text_model = text_model
        self._pipeline = None

    def _load_pipeline(self):
        cache = get_model_cache()
        lookup = cache_key("blip", self.caption_model)
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
                "Image summarization requires transformers. "
                "Install multimodal extras: uv sync --extra multimodal"
            ) from exc
        logger.info("Loading image captioning model: %s", self.caption_model)
        self._pipeline = pipeline("image-text-to-text", model=self.caption_model)
        cache.put(lookup, self._pipeline)
        return self._pipeline

    def _load_image(self, path: str | None, data: bytes | None):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Image summarization requires Pillow. "
                "Install multimodal extras: uv sync --extra multimodal"
            ) from exc
        if data is not None:
            return Image.open(BytesIO(data)).convert("RGB")
        if path is not None:
            return Image.open(path).convert("RGB")
        raise ValueError("Image input requires path or bytes")

    def caption(self, path: str | None = None, data: bytes | None = None) -> str:
        pipe = self._load_pipeline()
        image = self._load_image(path, data)
        try:
            result = pipe(image)
        except ValueError:
            # image-text-to-text pipelines (newer transformers) require a text prompt.
            result = pipe(image, text="")
        if isinstance(result, list) and result:
            return str(result[0].get("generated_text", "")).strip()
        if isinstance(result, dict):
            return str(result.get("generated_text", "")).strip()
        return str(result).strip()

    def summarize(
        self,
        path: str | None = None,
        data: bytes | None = None,
        max_length: int = 128,
        strategy: str = "stuff",
    ) -> dict[str, str]:
        caption = self.caption(path=path, data=data)
        summarizer = ModelFactory.create(self.text_model)
        summary = summarizer.summarize(caption, max_length=max_length, strategy=strategy)
        return {"caption": caption, "summary": summary, "model": self.text_model}
