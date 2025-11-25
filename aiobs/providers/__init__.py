"""Providers for LLM observability instrumentation.

Supported providers:
    - OpenAIProvider: Instruments OpenAI Chat Completions API
    - GeminiProvider: Instruments Google Gemini Generate Content API
"""

from .base import BaseProvider

__all__ = ["BaseProvider"]

# Lazy imports for optional providers
def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider
        return OpenAIProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider
        return GeminiProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

