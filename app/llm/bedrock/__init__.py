"""
Cliente para Amazon Bedrock con Claude 3.5 Sonnet.

Implementación vía HTTP con Authorization: Bearer (sin boto3/SigV4).

Variables de entorno soportadas:
  - AWS_REGION (por defecto: us-east-1)
  - AWS_BEARER_TOKEN_BEDROCK (token Bearer)
  - AWS_BEDROCK_MODEL_ID (por defecto: us.anthropic.claude-3-5-sonnet-20241022-v2:0)
  - BEDROCK_MAX_TOKENS (por defecto: 1024)
  - BEDROCK_TEMPERATURE (por defecto: 0.2)
  - BEDROCK_TOP_P (por defecto: 0.9)
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List

import requests

from ..base import BaseLLM
from ...config import settings

logger = logging.getLogger(__name__)

class BedrockLLM(BaseLLM):
    """
    Implementación de LLM usando Amazon Bedrock con Claude 3.5 Sonnet.

    Autenticación: Header Authorization: Bearer tomado de AWS_BEARER_TOKEN_BEDROCK.
    Región: Configurable vía AWS_REGION (por defecto "us-east-1").
    Modelo: por defecto "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Inicializa el cliente de Amazon Bedrock.
        
        Args:
            config: Configuración opcional para el cliente.
        """
        super().__init__(config)
        
        # Combinar configuración pasada con la del sistema
        cfg = {**(config or {}), **settings.provider_config()}
        
        # Configuración de entorno
        self.region = cfg.get("AWS_REGION") or os.environ.get("AWS_REGION") or settings.AWS_REGION
        
        # Modelo ID con soporte para múltiples nombres de variable
        self.model_id = (
            cfg.get("AWS_BEDROCK_MODEL_ID") or 
            cfg.get("BEDROCK_MODEL") or
            os.environ.get("AWS_BEDROCK_MODEL_ID") or
            os.environ.get("BEDROCK_MODEL") or
            settings.AWS_BEDROCK_MODEL_ID
        )
        
        # Token Bearer requerido
        self.bearer_token = (
            cfg.get("AWS_BEARER_TOKEN_BEDROCK") or 
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or 
            settings.AWS_BEARER_TOKEN_BEDROCK or 
            ""
        ).strip()
        
        if not self.bearer_token:
            raise RuntimeError(
                "AWS_BEARER_TOKEN_BEDROCK environment variable is required for Bedrock Bearer authentication"
            )
        
        # Configuración por defecto para inferencia
        max_tokens = cfg.get("BEDROCK_MAX_TOKENS") or os.environ.get("BEDROCK_MAX_TOKENS") or settings.BEDROCK_MAX_TOKENS
        temperature = cfg.get("BEDROCK_TEMPERATURE") or os.environ.get("BEDROCK_TEMPERATURE") or settings.BEDROCK_TEMPERATURE
        top_p = cfg.get("BEDROCK_TOP_P") or os.environ.get("BEDROCK_TOP_P") or settings.BEDROCK_TOP_P
        
        self.default_params = {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": float(top_p)
        }
        
        # Endpoint HTTP de Bedrock Runtime (converse)
        self.base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        # Formato: /model/{modelId}/converse
        self.converse_url = f"{self.base_url}/model/{self.model_id}/converse"
    
    def submit_prompt(self, prompt: str, **kwargs) -> str:
        """
        Envía un prompt al modelo Claude 3.5 Sonnet en Amazon Bedrock.
        
        Args:
            prompt: El mensaje del usuario.
            **kwargs: Parámetros adicionales para la inferencia.
            
        Returns:
            La respuesta generada por el modelo.
        """
        logger.info(f"BedrockLLM.submit_prompt called with prompt length: {len(prompt)}")
        try:
            # Construir mensajes y system blocks
            messages: List[Dict[str, Any]] = []
            system_blocks = None
            if self._system:
                system_blocks = [{"text": self._system}]

            for msg in self._history:
                role = msg.get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                content_text = msg.get("content", "")
                messages.append({
                    "role": role,
                    "content": [{"text": content_text}]
                })

            # Agregar el prompt actual como mensaje del usuario
            messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })

            # Configurar parámetros de inferencia
            inference_config = {**self.default_params}
            if "max_tokens" in kwargs:
                inference_config["maxTokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                inference_config["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                inference_config["topP"] = kwargs["top_p"]

            # Construir payload HTTP
            body: Dict[str, Any] = {
                "messages": messages,
            }
            if system_blocks is not None:
                body["system"] = system_blocks
            if inference_config:
                body["inferenceConfig"] = inference_config

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            }

            logger.info(f"Making request to Bedrock: {self.converse_url}")
            logger.info(f"Request body: {json.dumps(body, indent=2)}")

            resp = requests.post(
                self.converse_url,
                headers=headers,
                json=body,
                timeout=(10, 120),  # connect, read
            )

            logger.info(f"Bedrock response status: {resp.status_code}")

            if resp.status_code >= 400:
                # Intenta extraer mensaje de error del body
                err_text = resp.text
                logger.error(f"Bedrock error response: {err_text}")
                try:
                    err_json = resp.json()
                    err_text = err_json.get("message") or err_json.get("Message") or json.dumps(err_json)
                except Exception:
                    pass
                raise RuntimeError(f"Bedrock HTTP {resp.status_code}: {err_text}")

            data = resp.json()
            logger.info(f"Bedrock response data: {json.dumps(data, indent=2)}")
            
            response_message = (data.get("output", {}) or {}).get("message", {})
            blocks = response_message.get("content", []) or []
            response_text = ""
            for block in blocks:
                if isinstance(block, dict) and "text" in block:
                    response_text += block.get("text", "")

            logger.info(f"Bedrock response length: {len(response_text)}")
            self.assistant_message(response_text)
            return response_text

        except Exception as e:
            msg = str(e)
            logger.error("Error al invocar el modelo en Bedrock (Bearer): %s", msg, exc_info=True)
            raise RuntimeError(f"Bedrock invocation failed (Bearer): {msg}") from e