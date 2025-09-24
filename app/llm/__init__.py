"""
Simplified LLM package with direct provider implementations.

Available providers:
  - app.llm.gemini.GeminiLLM
  - app.llm.openai.OpenAILLM
  - app.llm.anthropic.AnthropicLLM
  - app.llm.bedrock.BedrockLLM
"""

import os
import logging
from typing import Optional, Dict, Any

from .base import BaseLLM  # noqa: F401

# Create independent logger to avoid circular imports
logger = logging.getLogger(__name__)


# Import providers
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
try:
    from .bedrock import BedrockLLM  # noqa: F401
except Exception:
    BedrockLLM = None  # optional


def create_llm(provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BaseLLM:
    """
    Factory function to create LLM instances based on provider.
    
    Args:
        provider: LLM provider name (google, openai, anthropic, bedrock)
        config: Optional configuration dictionary
    
    Returns:
        BaseLLM instance
    """
    from ..config import settings
    
    prov = (provider or settings.LLM_PROVIDER or os.environ.get("LLM_PROVIDER") or "bedrock").lower()
    cfg = config or {}

    logger.info("Using LLM provider: %s", prov)

    if prov == "google" or prov == "gemini":
        if GeminiLLM is None:
            raise ImportError("GeminiLLM not available")
        return GeminiLLM(cfg)
    elif prov == "openai":
        if OpenAILLM is None:
            raise ImportError("OpenAILLM not available")
        return OpenAILLM(cfg)
    elif prov == "anthropic":
        if AnthropicLLM is None:
            raise ImportError("AnthropicLLM not available")
        return AnthropicLLM(cfg)
    elif prov == "bedrock":
        if BedrockLLM is None:
            raise ImportError("BedrockLLM not available")
        return BedrockLLM(cfg)
    else:
        raise ValueError(f"Unsupported LLM provider: {prov}")


def create_llm_from_env(config: Optional[Dict[str, Any]] = None) -> BaseLLM:
    """Create LLM from environment configuration."""
    return create_llm(config=config)
