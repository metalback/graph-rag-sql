import os
from typing import Any, Dict, Optional
from langchain_google_genai import GoogleGenerativeAI
from ..domain.interfaces import BaseLLMConnector
from ...config import settings


class GoogleLLMConnector(BaseLLMConnector):
    """
    Google Generative AI connector using env vars (.env) or passed config.
    Required: GOOGLE_AI_KEY
    Optional: GOOGLE_MODEL, GOOGLE_MAX_TOKENS, GOOGLE_TEMPERATURE, GOOGLE_TOP_P, GOOGLE_TOP_K
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**(config or {}), **settings.provider_config()}
        api_key = cfg.get("GOOGLE_AI_KEY") or os.environ.get("GOOGLE_AI_KEY") or settings.GOOGLE_AI_KEY
        if not api_key:
            raise ValueError("GOOGLE_AI_KEY environment variable is not set")

        self.api_key = api_key
        self.model = cfg.get("GOOGLE_MODEL") or os.environ.get("GOOGLE_MODEL") or settings.GOOGLE_MODEL
        self.max_output_tokens = int(
            cfg.get("GOOGLE_MAX_TOKENS") or os.environ.get("GOOGLE_MAX_TOKENS") or settings.GOOGLE_MAX_TOKENS
        )
        self.temperature = float(
            cfg.get("GOOGLE_TEMPERATURE") or os.environ.get("GOOGLE_TEMPERATURE") or settings.GOOGLE_TEMPERATURE
        )
        self.top_p = float(cfg.get("GOOGLE_TOP_P") or os.environ.get("GOOGLE_TOP_P") or settings.GOOGLE_TOP_P)
        self.top_k = int(cfg.get("GOOGLE_TOP_K") or os.environ.get("GOOGLE_TOP_K") or settings.GOOGLE_TOP_K)

    def get_llm(self) -> Any:
        return GoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
