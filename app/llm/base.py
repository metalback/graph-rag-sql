from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):
    """
    Interface for LLM implementations.

    Required methods per spec:
      - __init__(self, config=None)
      - system_message(self, message: str) -> Any
      - user_message(self, message: str) -> Any
      - assistant_message(self, message: str) -> Any
      - submit_prompt(self, prompt, **kwargs) -> str
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._system: Optional[str] = None
        self._history: List[Dict[str, str]] = []  # list of {role, content}

    def system_message(self, message: str) -> Any:
        self._system = message
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        self._history.append({"role": "user", "content": message})
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> Any:
        self._history.append({"role": "assistant", "content": message})
        return {"role": "assistant", "content": message}

    @abstractmethod
    def submit_prompt(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    # Helpers for implementations
    def _build_prompt(self, prompt: str) -> str:
        parts: List[str] = []
        if self._system:
            parts.append(f"[SYSTEM]\n{self._system}")
        for msg in self._history:
            parts.append(f"[{msg['role'].upper()}]\n{msg['content']}")
        parts.append(f"[USER]\n{prompt}")
        return "\n\n".join(parts)
