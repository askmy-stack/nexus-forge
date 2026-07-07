"""Singleton LRU cache for heavy model instances."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class ModelCache:
    """Thread-safe LRU cache for summarizer and multimodal model instances."""

    def __init__(self, max_size: int = 8):
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())


_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    return _model_cache


def cache_key(*parts: str) -> str:
    return ":".join(parts)
