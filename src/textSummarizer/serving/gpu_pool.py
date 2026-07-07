"""GPU inference pool with concurrency limiting and TTL model cache."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")


class GPUJobPool:
    """Limit concurrent GPU-bound inference and cache loaded models with TTL eviction."""

    def __init__(
        self,
        max_jobs: int | None = None,
        cache_ttl_seconds: int | None = None,
        max_cached_models: int = 4,
    ):
        self.max_jobs = max_jobs or int(os.getenv("MAX_GPU_JOBS", "2"))
        self.cache_ttl_seconds = cache_ttl_seconds or int(
            os.getenv("MODEL_CACHE_TTL_SECONDS", "3600")
        )
        self.max_cached_models = max_cached_models
        self._semaphore = threading.BoundedSemaphore(self.max_jobs)
        self._model_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @contextmanager
    def acquire(self):
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, (_, loaded_at) in self._model_cache.items()
            if now - loaded_at > self.cache_ttl_seconds
        ]
        for key in expired:
            del self._model_cache[key]

    def _trim_cache(self) -> None:
        while len(self._model_cache) > self.max_cached_models:
            self._model_cache.popitem(last=False)

    def get_model(self, model_name: str, loader: Callable[[], Any]) -> Any:
        with self._cache_lock:
            self._evict_expired()
            cached = self._model_cache.get(model_name)
            if cached is not None:
                model, _ = cached
                self._model_cache.move_to_end(model_name)
                self._model_cache[model_name] = (model, time.monotonic())
                return model

        with self.acquire():
            with self._cache_lock:
                cached = self._model_cache.get(model_name)
                if cached is not None:
                    model, _ = cached
                    return model

            model = loader()
            with self._cache_lock:
                self._model_cache[model_name] = (model, time.monotonic())
                self._trim_cache()
            return model

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self.acquire():
            return fn(*args, **kwargs)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._model_cache.clear()


_pool: GPUJobPool | None = None
_pool_lock = threading.Lock()


def get_gpu_pool() -> GPUJobPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = GPUJobPool()
        return _pool
