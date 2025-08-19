"""
Compatibility shim for legacy imports.

This module re-exports the Clean Architecture components from app.llm.*
Prefer importing from app.llm going forward:
    from .llm import LLMConnectorFactory, BaseLLMConnector
"""

from .llm.domain.interfaces import BaseLLMConnector
from .llm.application.factory import LLMConnectorFactory
from .llm.infrastructure.google_connector import GoogleLLMConnector as LLMConnector  # backward-compat alias
from .llm.infrastructure.google_connector import GoogleLLMConnector
try:
    from .llm.infrastructure.openai_connector import OpenAILLMConnector  # type: ignore
except Exception:  # pragma: no cover
    OpenAILLMConnector = None  # type: ignore
try:
    from .llm.infrastructure.anthropic_connector import AnthropicLLMConnector  # type: ignore
except Exception:  # pragma: no cover
    AnthropicLLMConnector = None  # type: ignore

