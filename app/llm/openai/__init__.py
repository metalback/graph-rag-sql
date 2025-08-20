from __future__ import annotations
from typing import Any, Dict, Optional
from ..base import BaseLLM
from ...config import settings

try:
    from langchain_openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAILLM(BaseLLM):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = {**(config or {}), **settings.provider_config()}
        self._api_key = cfg.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        self._model = cfg.get("OPENAI_MODEL") or settings.OPENAI_MODEL
        self._max_tokens = cfg.get("OPENAI_MAX_TOKENS") or settings.OPENAI_MAX_TOKENS
        self._temperature = float(cfg.get("OPENAI_TEMPERATURE") or settings.OPENAI_TEMPERATURE)
        self._top_p = cfg.get("OPENAI_TOP_P") or settings.OPENAI_TOP_P

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        if OpenAI is None:
            raise ImportError("langchain-openai is not installed")
        llm = OpenAI(
            api_key=self._api_key,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        full_prompt = self._build_prompt(prompt)
        return llm.invoke(full_prompt)  # type: ignore
