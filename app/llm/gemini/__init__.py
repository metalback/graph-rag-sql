from __future__ import annotations
from typing import Any, Dict, Optional
from ..base import BaseLLM
from ...config import settings

try:
    from langchain_google_genai import GoogleGenerativeAI
except Exception:  # pragma: no cover
    GoogleGenerativeAI = None  # type: ignore

# Native SDK for tool/function calling
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


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

    # Tool/function calling via native Gemini SDK
    def function_call(
        self,
        system_and_context: str,
        user_question: str,
        tools: Any,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke Gemini with function declarations and extract the first function call.

        Returns {"name": str, "args": dict} on success, raises informative errors otherwise.
        """
        if genai is None:
            raise ImportError("google-generativeai is not available. Ensure the package is installed.")
        if not self._api_key:
            raise RuntimeError("GOOGLE_AI_KEY is not configured.")

        genai.configure(api_key=self._api_key)
        cfg = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if generation_config:
            cfg.update(generation_config)

        model = genai.GenerativeModel(
            model_name=self._model,
            tools=tools,
            generation_config=cfg,
        )

        resp = model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": f"{system_and_context}\n\n# PREGUNTA\n{user_question}"}],
            }]
        )

        # Extract function_call
        try:
            parts = resp.candidates[0].content.parts  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Gemini response missing candidates/parts: {e}")

        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc and getattr(fc, "name", None):
                return {"name": fc.name, "args": getattr(fc, "args", {})}

        raise RuntimeError("No function call produced by the model.")
