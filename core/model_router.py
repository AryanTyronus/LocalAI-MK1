"""
Model Router for SYNAPSE - Routes between fast/smart models.
Extensible design for future local models.
"""

import sys
sys.path.insert(0, '..')
from models.openrouter import OpenRouterModel
from models.qwen_local import QwenLocalModel
from typing import List, Dict, Any


class ModelRouter:
    def __init__(self):
        """
        Initialize model router.
        smart_model: OpenRouter Nemotron (smart)
        fast_model: Placeholder for future local model
        """
        self.smart_model = OpenRouterModel("nvidia/nemotron-3-super-120b-a12b:free")
        self.fast_model = QwenLocalModel()
        
        print("ModelRouter initialized: smart_model (Nemotron) + fast_model (Qwen3) ready")
    
    def generate(self, task_type: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Route request to appropriate model.
        
        Args:
            task_type: "analysis", "reasoning", "fast", "simple"
            messages: Chat messages
            
        Returns:
            Model response dict or error dict
        """
        task_type = task_type.lower().strip()
        
        if task_type in ("analysis", "reasoning"):
            model = self.smart_model
            model_name = "🧠 Nemotron (smart)"
        elif task_type in ("fast", "simple"):
            model = self.fast_model
            model_name = "⚡ Qwen3 (fast)"
        else:
            model = self.smart_model
            model_name = "smart (default)"
        
        print(f"📡 Routing '{task_type}' to {model_name}")
        
        try:
            return model.generate(messages)
        except Exception as e:
            return {
                "error": {
                    "message": f"Model generation failed: {str(e)}",
                    "model_used": model_name,
                    "task_type": task_type
                }
            }
