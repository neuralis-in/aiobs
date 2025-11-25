from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseGeminiRequest(BaseModel):
    """Base class for Gemini request capture models."""

    model: Optional[str] = Field(default=None)

    def redacted(self) -> "BaseGeminiRequest":
        """Return a copy safe for logging (override in subclasses)."""
        return self


class BaseGeminiResponse(BaseModel):
    """Base class for Gemini response capture models."""

    model: Optional[str] = Field(default=None)
    usage: Optional[Dict[str, Any]] = Field(default=None)

    def redacted(self) -> "BaseGeminiResponse":
        """Return a copy safe for logging (override in subclasses)."""
        return self

