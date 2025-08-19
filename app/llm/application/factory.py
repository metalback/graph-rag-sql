import os
from typing import Optional, Dict, Any
from ..domain.interfaces import BaseLLMConnector
from ..infrastructure.google_connector import GoogleLLMConnector
try:
    from ..infrastructure.openai_connector import OpenAILLMConnector  # type: ignore
except Exception:  # pragma: no cover
    OpenAILLMConnector = None  # type: ignore
try:
    from ..infrastructure.anthropic_connector import AnthropicLLMConnector  # type: ignore
except Exception:  # pragma: no cover
    AnthropicLLMConnector = None  # type: ignore
from ...config import settings


class LLMConnectorFactory:
    """Application factory to create LLM connectors based on configuration.

    Precedence: explicit provider arg > settings.LLM_PROVIDER > env var.
    """

    @staticmethod
    def create(provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BaseLLMConnector:
        prov = (provider or settings.LLM_PROVIDER or os.environ.get("LLM_PROVIDER") or "google").lower()
        cfg = config or {}

        if prov == "google":
            return GoogleLLMConnector(cfg)
        if prov == "openai":
            if OpenAILLMConnector is None:
                raise ImportError("langchain-openai not installed; cannot use OpenAI provider")
            return OpenAILLMConnector(cfg)  # type: ignore
        if prov == "anthropic":
            if AnthropicLLMConnector is None:
                raise ImportError("langchain-anthropic not installed; cannot use Anthropic provider")
            return AnthropicLLMConnector(cfg)  # type: ignore
        raise ValueError(f"Unsupported LLM provider: {prov}")

    @staticmethod
    def from_env_or_config(config: Optional[Dict[str, Any]] = None) -> BaseLLMConnector:
        return LLMConnectorFactory.create(config=config)
