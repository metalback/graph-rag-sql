from abc import ABC, abstractmethod
from typing import Any

class BaseLLMConnector(ABC):
    """
    Domain interface for LLM connectors. Implementations must provide
    a get_llm() method returning a LangChain-compatible LLM instance.
    """

    @abstractmethod
    def get_llm(self) -> Any:
        """Return an initialized LLM client compatible with LangChain."""
        raise NotImplementedError
