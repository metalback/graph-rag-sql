from __future__ import annotations
from typing import Any, Dict, Optional
from ..base import BaseLLM
from ...config import settings

try:
    from langchain_google_genai import GoogleGenerativeAI
except Exception:  # pragma: no cover
    GoogleGenerativeAI = None  # type: ignore


class GeminiLLM(BaseLLM):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = {**(config or {}), **settings.provider_config()}
        self._api_key = cfg.get("GOOGLE_AI_KEY") or settings.GOOGLE_AI_KEY
        self._model = cfg.get("GOOGLE_MODEL") or settings.GOOGLE_MODEL
        self._max_output_tokens = int(cfg.get("GOOGLE_MAX_TOKENS") or settings.GOOGLE_MAX_TOKENS)
        self._temperature = float(cfg.get("GOOGLE_TEMPERATURE") or settings.GOOGLE_TEMPERATURE)
        self._top_p = float(cfg.get("GOOGLE_TOP_P") or settings.GOOGLE_TOP_P)
        self._top_k = int(cfg.get("GOOGLE_TOP_K") or settings.GOOGLE_TOP_K)

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        if GoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed")
        llm = GoogleGenerativeAI(
            google_api_key=self._api_key,
            model=self._model,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
        )
        full_prompt = self._build_prompt(prompt)
        # minimal generation using LangChain LLM interface
        return llm.invoke(full_prompt)  # type: ignore
