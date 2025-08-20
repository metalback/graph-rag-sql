from __future__ import annotations
from typing import Any, Dict, Optional
from ..base import BaseLLM
from ...config import settings

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore


class AnthropicLLM(BaseLLM):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = {**(config or {}), **settings.provider_config()}
        self._api_key = cfg.get("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
        self._model = cfg.get("ANTHROPIC_MODEL") or settings.ANTHROPIC_MODEL
        self._max_tokens = int(cfg.get("ANTHROPIC_MAX_TOKENS") or settings.ANTHROPIC_MAX_TOKENS)
        self._temperature = float(cfg.get("ANTHROPIC_TEMPERATURE") or settings.ANTHROPIC_TEMPERATURE)

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic is not installed")
        llm = ChatAnthropic(
            api_key=self._api_key,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        full_prompt = self._build_prompt(prompt)
        # ChatAnthropic returns AIMessage; extract content
        result = llm.invoke(full_prompt)  # type: ignore
        try:
            return result.content  # type: ignore[attr-defined]
        except Exception:
            return str(result)
