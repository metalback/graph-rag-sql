import os
from typing import Any, Dict, Optional
from ..domain.interfaces import BaseLLMConnector
from ...config import settings

try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception as e:  # pragma: no cover
    ChatAnthropic = None  # type: ignore


class AnthropicLLMConnector(BaseLLMConnector):
    """
    Anthropic connector using env vars (.env) or passed config.
    Required: ANTHROPIC_API_KEY
    Optional: ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS, ANTHROPIC_TEMPERATURE
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**(config or {}), **settings.provider_config()}
        api_key = cfg.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic is not installed. Add 'langchain-anthropic' to requirements.")

        self.api_key = api_key
        self.model = cfg.get("ANTHROPIC_MODEL") or os.environ.get("ANTHROPIC_MODEL") or settings.ANTHROPIC_MODEL
        self.max_tokens = int(cfg.get("ANTHROPIC_MAX_TOKENS") or os.environ.get("ANTHROPIC_MAX_TOKENS") or settings.ANTHROPIC_MAX_TOKENS)
        self.temperature = float(cfg.get("ANTHROPIC_TEMPERATURE") or os.environ.get("ANTHROPIC_TEMPERATURE") or settings.ANTHROPIC_TEMPERATURE)

    def get_llm(self) -> Any:
        return ChatAnthropic(  # type: ignore
            api_key=self.api_key,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
