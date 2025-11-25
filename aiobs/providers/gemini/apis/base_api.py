from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseGeminiAPIModule(ABC):
    """Abstract interface for a Gemini API module."""

    name: str = "gemini-api"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        """Install instrumentation and return optional unpatch function."""
        raise NotImplementedError

