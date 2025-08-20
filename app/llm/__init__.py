# Clean Architecture package for LLM connectors
"""
Convenience exports for the new modular LLM interface.

Use BaseLLM with provider implementations in subpackages:
  - app.llm.gemini.GeminiLLM
  - app.llm.openai.OpenAILLM
  - app.llm.anthropic.AnthropicLLM
"""

from .base import BaseLLM  # noqa: F401
try:
    from .gemini import GeminiLLM  # noqa: F401
except Exception:
    GeminiLLM = None  # optional
try:
    from .openai import OpenAILLM  # noqa: F401
except Exception:
    OpenAILLM = None  # optional
try:
    from .anthropic import AnthropicLLM  # noqa: F401
except Exception:
    AnthropicLLM = None  # optional
