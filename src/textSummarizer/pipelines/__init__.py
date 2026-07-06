"""Long-document summarization strategies."""

STRATEGIES: tuple[str, ...] = ("stuff", "map_reduce", "refine", "hierarchical", "rag")
STRATEGY_PATTERN = "^(stuff|map_reduce|refine|hierarchical|rag)$"

__all__ = ["STRATEGIES", "STRATEGY_PATTERN"]
