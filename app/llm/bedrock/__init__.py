from __future__ import annotations
import os
from typing import Any, Dict, Optional
from ..base import BaseLLM
from ...config import settings

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore


class BedrockLLM(BaseLLM):
    """
    AWS Bedrock LLM implementation using boto3.
    
    Required environment variables:
    - AWS_BEARER_TOKEN_BEDROCK: API key for Bedrock
    
    Optional environment variables:
    - AWS_REGION: AWS region (default: us-east-1)
    - BEDROCK_MODEL: Model ID (default: us.anthropic.claude-3-5-haiku-20241022-v1:0)
    - BEDROCK_MAX_TOKENS: Max tokens (default: 1024)
    - BEDROCK_TEMPERATURE: Temperature (default: 0.2)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = {**(config or {}), **settings.provider_config()}
        
        # Set up AWS credentials
        self._api_key = cfg.get("AWS_BEARER_TOKEN_BEDROCK") or settings.AWS_BEARER_TOKEN_BEDROCK
        if not self._api_key:
            raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable is required")
            
        # Set the environment variable for boto3
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self._api_key
        
        # Configuration
        self._region = cfg.get("AWS_REGION") or settings.AWS_REGION
        self._model = cfg.get("BEDROCK_MODEL") or settings.BEDROCK_MODEL
        self._max_tokens = int(cfg.get("BEDROCK_MAX_TOKENS") or settings.BEDROCK_MAX_TOKENS)
        self._temperature = float(cfg.get("BEDROCK_TEMPERATURE") or settings.BEDROCK_TEMPERATURE)
        
        if boto3 is None:
            raise ImportError("boto3 is not installed. Install it with: pip install boto3")
            
        # Initialize Bedrock client
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self._region
        )

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        """Submit a prompt to AWS Bedrock and return the response."""
        try:
            # Build the full prompt with system message and history
            full_prompt = self._build_prompt(prompt)
            
            # Prepare messages for Bedrock converse API
            messages = [{"role": "user", "content": [{"text": full_prompt}]}]
            
            # Call Bedrock converse API
            response = self._client.converse(
                modelId=self._model,
                messages=messages,
                inferenceConfig={
                    'maxTokens': self._max_tokens,
                    'temperature': self._temperature,
                }
            )
            
            # Extract the response content
            if 'output' in response and 'message' in response['output']:
                content = response['output']['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get('text', '')
                elif isinstance(content, str):
                    return content
            
            return str(response)
            
        except Exception as e:
            raise RuntimeError(f"Error calling AWS Bedrock: {str(e)}")
