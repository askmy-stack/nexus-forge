import base64
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InputType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalInput:
    """Unified input for multimodal summarization."""

    input_type: InputType
    path: str | None = None
    data: bytes | None = None
    base64_data: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_bytes(self) -> bytes | None:
        if self.data is not None:
            return self.data
        if self.base64_data is not None:
            payload = self.base64_data
            if "," in payload:
                payload = payload.split(",", 1)[1]
            return base64.b64decode(payload)
        return None

    def resolve_text(self) -> str | None:
        if self.text is not None:
            return self.text
        if self.input_type == InputType.TEXT:
            raw = self.resolve_bytes()
            if raw is not None:
                return raw.decode("utf-8")
        return None
