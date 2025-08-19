import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional at runtime; if missing, env must be provided by the environment
    pass


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(',') if p.strip()]
    return parts or None


@dataclass
class Settings:
    # LLM Provider selection
    LLM_PROVIDER: str = os.environ.get('LLM_PROVIDER', 'google').lower()

    # CORS
    CORS_ORIGINS_RAW: Optional[str] = os.environ.get('CORS_ORIGINS')

    # Google config
    GOOGLE_AI_KEY: Optional[str] = os.environ.get('GOOGLE_AI_KEY')
    GOOGLE_MODEL: str = os.environ.get('GOOGLE_MODEL', 'gemini-1.5-pro')
    GOOGLE_MAX_TOKENS: int = int(os.environ.get('GOOGLE_MAX_TOKENS', '1024'))
    GOOGLE_TEMPERATURE: float = float(os.environ.get('GOOGLE_TEMPERATURE', '0.2'))
    GOOGLE_TOP_P: float = float(os.environ.get('GOOGLE_TOP_P', '0.8'))
    GOOGLE_TOP_K: int = int(os.environ.get('GOOGLE_TOP_K', '40'))

    # OpenAI config
    OPENAI_API_KEY: Optional[str] = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS: Optional[int] = (
        int(os.environ['OPENAI_MAX_TOKENS']) if os.environ.get('OPENAI_MAX_TOKENS') else None
    )
    OPENAI_TEMPERATURE: float = float(os.environ.get('OPENAI_TEMPERATURE', '0.2'))
    OPENAI_TOP_P: Optional[float] = (
        float(os.environ['OPENAI_TOP_P']) if os.environ.get('OPENAI_TOP_P') else None
    )

    # Anthropic config
    ANTHROPIC_API_KEY: Optional[str] = os.environ.get('ANTHROPIC_API_KEY')
    ANTHROPIC_MODEL: str = os.environ.get('ANTHROPIC_MODEL', 'claude-3-5-sonnet-latest')
    ANTHROPIC_MAX_TOKENS: int = int(os.environ.get('ANTHROPIC_MAX_TOKENS', '1024'))
    ANTHROPIC_TEMPERATURE: float = float(os.environ.get('ANTHROPIC_TEMPERATURE', '0.2'))

    @property
    def CORS_ORIGINS(self) -> Optional[List[str]]:
        return _split_csv(self.CORS_ORIGINS_RAW)

    def provider_config(self) -> Dict[str, Any]:
        if self.LLM_PROVIDER == 'google':
            return {
                'GOOGLE_AI_KEY': self.GOOGLE_AI_KEY,
                'GOOGLE_MODEL': self.GOOGLE_MODEL,
                'GOOGLE_MAX_TOKENS': self.GOOGLE_MAX_TOKENS,
                'GOOGLE_TEMPERATURE': self.GOOGLE_TEMPERATURE,
                'GOOGLE_TOP_P': self.GOOGLE_TOP_P,
                'GOOGLE_TOP_K': self.GOOGLE_TOP_K,
            }
        if self.LLM_PROVIDER == 'openai':
            return {
                'OPENAI_API_KEY': self.OPENAI_API_KEY,
                'OPENAI_MODEL': self.OPENAI_MODEL,
                'OPENAI_MAX_TOKENS': self.OPENAI_MAX_TOKENS,
                'OPENAI_TEMPERATURE': self.OPENAI_TEMPERATURE,
                'OPENAI_TOP_P': self.OPENAI_TOP_P,
            }
        if self.LLM_PROVIDER == 'anthropic':
            return {
                'ANTHROPIC_API_KEY': self.ANTHROPIC_API_KEY,
                'ANTHROPIC_MODEL': self.ANTHROPIC_MODEL,
                'ANTHROPIC_MAX_TOKENS': self.ANTHROPIC_MAX_TOKENS,
                'ANTHROPIC_TEMPERATURE': self.ANTHROPIC_TEMPERATURE,
            }
        return {}


# Singleton-like settings instance
settings = Settings()
