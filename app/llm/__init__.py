# Clean Architecture package for LLM connectors
# Expose convenient imports
from .domain.interfaces import BaseLLMConnector
from .application.factory import LLMConnectorFactory
from .infrastructure.google_connector import GoogleLLMConnector
try:
    from .infrastructure.openai_connector import OpenAILLMConnector
except Exception:
    OpenAILLMConnector = None  # optional
try:
    from .infrastructure.anthropic_connector import AnthropicLLMConnector
except Exception:
    AnthropicLLMConnector = None  # optional
