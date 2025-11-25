from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .base import BaseGeminiRequest, BaseGeminiResponse


class ContentPart(BaseModel):
    """A part of content (text, image, etc.)."""
    text: Optional[str] = Field(default=None)
    inline_data: Optional[Dict[str, Any]] = Field(default=None)


class Content(BaseModel):
    """Content structure for Gemini messages."""
    role: Optional[str] = Field(default=None)
    parts: Optional[List[ContentPart]] = Field(default=None)


class GenerateContentRequest(BaseGeminiRequest):
    """Request model for generate_content API."""
    contents: Optional[Union[str, List[Content], Any]] = Field(default=None)
    system_instruction: Optional[Any] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    other: Dict[str, Any] = Field(default_factory=dict)


class GenerateContentResponse(BaseGeminiResponse):
    """Response model for generate_content API."""
    text: Optional[str] = Field(default=None)
    candidates: Optional[List[Dict[str, Any]]] = Field(default=None)

