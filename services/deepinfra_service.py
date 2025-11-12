"""DeepInfra API service for accessing various models."""

import os
import requests
from typing import Dict, Any, Optional, List

class DeepInfraService:
    """Service for interacting with DeepInfra API."""
    
    BASE_URL = "https://api.deepinfra.com/v1/openai"
    
    # DeepInfra models as of 2025
    MODELS = {
        # Chat/Text models
        "meta-llama/Meta-Llama-3.1-405B-Instruct": {"type": "text", "context": 32768},
        "meta-llama/Meta-Llama-3.1-70B-Instruct": {"type": "text", "context": 131072},
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {"type": "text", "context": 131072},
        "meta-llama/Llama-3.2-90B-Vision-Instruct": {"type": "vision", "context": 131072},
        "meta-llama/Llama-3.2-11B-Vision-Instruct": {"type": "vision", "context": 131072},
        "Qwen/Qwen2.5-72B-Instruct": {"type": "text", "context": 32768},
        "Qwen/Qwen2.5-Coder-32B-Instruct": {"type": "text", "context": 32768},
        "Qwen/QwQ-32B-Preview": {"type": "text", "context": 32768},
        "google/gemma-2-27b-it": {"type": "text", "context": 8192},
        "microsoft/WizardLM-2-8x22B": {"type": "text", "context": 65536},
        "mistralai/Mixtral-8x22B-Instruct-v0.1": {"type": "text", "context": 65536},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"type": "text", "context": 32768},
        "mistralai/Mistral-7B-Instruct-v0.3": {"type": "text", "context": 32768},
        "nvidia/Llama-3.1-Nemotron-70B-Instruct": {"type": "text", "context": 131072},
        "cognitivecomputations/dolphin-2.9.1-llama-3-70b": {"type": "text", "context": 8192},
        "Gryphe/MythoMax-L2-13b": {"type": "text", "context": 4096},
        "openchat/openchat-3.6-8b": {"type": "text", "context": 8192},
        # Code models
        "deepseek-ai/DeepSeek-Coder-V2-Instruct": {"type": "text", "context": 163840},
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {"type": "text", "context": 163840},
        # Embedding models
        "BAAI/bge-large-en-v1.5": {"type": "embedding", "dimensions": 1024},
        "BAAI/bge-base-en-v1.5": {"type": "embedding", "dimensions": 768},
        "sentence-transformers/all-MiniLM-L6-v2": {"type": "embedding", "dimensions": 384},
        "thenlper/gte-large": {"type": "embedding", "dimensions": 1024},
        "intfloat/e5-large-v2": {"type": "embedding", "dimensions": 1024},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepInfra service."""
        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError("DeepInfra API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def generate_embedding(
        self,
        model: str,
        text: str
    ) -> List[float]:
        """Generate embedding for text."""
        url = f"{self.BASE_URL}/embeddings"
        
        payload = {
            "model": model,
            "input": text
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]
    
    @classmethod
    def get_text_models(cls) -> List[str]:
        """Get list of available text models."""
        return [model for model, info in cls.MODELS.items() if info["type"] == "text"]
    
    @classmethod
    def get_embedding_models(cls) -> List[str]:
        """Get list of available embedding models."""
        return [model for model, info in cls.MODELS.items() if info["type"] == "embedding"]
    
    @classmethod
    def get_vision_models(cls) -> List[str]:
        """Get list of available vision models."""
        return [model for model, info in cls.MODELS.items() if info["type"] == "vision"]
