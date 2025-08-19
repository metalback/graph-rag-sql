import os
from typing import Any, Dict, Optional
from ..domain.interfaces import BaseLLMConnector
from ...config import settings

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


class OpenAILLMConnector(BaseLLMConnector):
    """
    OpenAI connector using env vars (.env) or passed config.
    Required: OPENAI_API_KEY
    Optional: OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE, OPENAI_TOP_P
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**(config or {}), **settings.provider_config()}
        api_key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Add 'langchain-openai' to requirements.")

        self.api_key = api_key
        self.model = cfg.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or settings.OPENAI_MODEL
        self.max_tokens = cfg.get("OPENAI_MAX_TOKENS") or os.environ.get("OPENAI_MAX_TOKENS") or settings.OPENAI_MAX_TOKENS
        self.temperature = float(cfg.get("OPENAI_TEMPERATURE") or os.environ.get("OPENAI_TEMPERATURE") or settings.OPENAI_TEMPERATURE)
        self.top_p = cfg.get("OPENAI_TOP_P") or os.environ.get("OPENAI_TOP_P") or settings.OPENAI_TOP_P

    def get_llm(self) -> Any:
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = int(self.max_tokens)
        if self.top_p is not None:
            params["top_p"] = float(self.top_p)
        return ChatOpenAI(**params)  # type: ignore
