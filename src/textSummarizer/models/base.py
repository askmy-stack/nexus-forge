from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, max_length: int = 128, strategy: str = "stuff") -> str:
        """Summarize input text."""
