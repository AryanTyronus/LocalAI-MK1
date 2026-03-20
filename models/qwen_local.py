"""
Qwen Local Model - Ollama Qwen3 8B integration.
Fast local model for ModelRouter.
"""

import json
import requests
from typing import List, Dict, Any


class QwenLocalModel:
    def __init__(self):
        """
        Initialize Qwen3 8B local model via Ollama.
        """
        self.model_name = "qwen3:8b"
        self.base_url = "http://localhost:11434/api/generate"
        print("⚡ Qwen3 8B (fast model) initialized")
    
    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate response from Qwen local model.
        
        Args:
            messages: OpenAI chat format messages
            
        Returns:
            OpenAI-like response dict or error
        """
        # Convert chat messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        prompt = "\\n".join(prompt_parts)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=120,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("response", "")
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content
                        }
                    }]
                }
            else:
                return {
                    "error": {
                        "message": f"Ollama error {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                }
        except requests.exceptions.ConnectionError:
            return {
                "error": {
                    "message": "Ollama not running at http://localhost:11434. Start with 'ollama serve' and pull 'qwen3:8b'.",
                    "status_code": None
                }
            }
        except Exception as e:
            return {
                "error": {
                    "message": f"Generation failed: {str(e)}",
                    "status_code": None
                }
            }
