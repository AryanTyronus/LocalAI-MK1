"""
OpenRouter Model Integration - Simple requests-based client.
Production-ready with error handling and environment variable loading.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
from typing import List, Dict, Any, Optional
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables at module level if dotenv available, else use os.getenv directly
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OpenRouterModel:
    def __init__(self, model_name: str = "nvidia/nemotron-3-super-120b-a12b:free"):
        """
        Initialize OpenRouter model.
        
        Args:
            model_name: OpenRouter model identifier
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment. Set in .env file.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Session with retry and timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate response from OpenRouter.
        
        Args:
            messages: List of OpenAI-format chat messages
            
        Returns:
            Full API response JSON or error dict
            
        Raises:
            ValueError: On API key validation failure (already checked in init)
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": {
                        "message": f"API request failed: {response.text}",
                        "status_code": response.status_code,
                    }
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "error": {
                    "message": f"Request failed: {str(e)}",
                    "status_code": getattr(e.response, 'status_code', None),
                }
            }
        except Exception as e:
            return {
                "error": {
                    "message": f"Unexpected error: {str(e)}",
                    "status_code": None,
                }
            }
