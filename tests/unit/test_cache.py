"""Tests for model cache."""

from textSummarizer.models import ModelFactory
from textSummarizer.models.cache import ModelCache, get_model_cache


def test_model_cache_lru_eviction():
    cache = ModelCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_model_factory_uses_cache():
    cache = get_model_cache()
    cache.clear()
    first = ModelFactory.create("extractive")
    second = ModelFactory.create("extractive")
    assert first is second
