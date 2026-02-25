import os
import numpy as np
from sentence_transformers import SentenceTransformer
from mlx_lm import generate as mlx_generate
from core.config import MODEL_NAME
from core.logger import logger
from core.chat_formatter import ChatFormatter, Message, Role, ModelType

# If running in dev mode, a lightweight fake model will be used to avoid
# downloading large models and consuming excessive resources.


class FakeModelManager:
    def __init__(self):
        logger.info("Using FakeModelManager (LOCALAI_DEV_MODE=1)")
        self.tokenizer = None
        self._embed_dim = 8

    def generate(self, prompt, max_tokens=300, temperature: float = 0.7):
        # Simple deterministic behavior for dev mode
        if 'OPEN_APP_TEST' in prompt:
            return '{"action": "open_app", "parameters": {"app_name": "Safari"}}'
        if 'CODE_TEST' in prompt:
            return 'def hello():\n    return "hello"'
        return 'This is a fake model response.'

    def format_chat(self, system: str, user: str, history: str = None) -> str:
        return (system or '') + "\n\n" + (user or '')

    def format_messages(self, messages: list) -> str:
        return '\n'.join([f"{m.role}: {m.content}" for m in messages])

    def embed(self, texts):
        return np.zeros((len(texts), self._embed_dim), dtype=float)


class ModelManager:
    _instance = None

    def __init__(self):
        logger.info("Loading MLX model...")
        # Real model loading path
        from mlx_lm import load
        self.model, self.tokenizer = load(MODEL_NAME)

        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize chat formatter for Qwen2.5
        # Detect model type from MODEL_NAME
        if "qwen2.5" in MODEL_NAME.lower() or "qwen" in MODEL_NAME.lower():
            self.chat_formatter = ChatFormatter(ModelType.QWEN2_5)
        else:
            self.chat_formatter = ChatFormatter(ModelType.DEFAULT)

    @classmethod
    def get_instance(cls):
        # If dev mode enabled, return a lightweight fake manager to avoid heavy model loads
        if cls._instance is None:
            if os.getenv('LOCALAI_DEV_MODE') == '1':
                cls._instance = FakeModelManager()
            else:
                cls._instance = ModelManager()
        return cls._instance

    def generate(self, prompt, max_tokens=300, temperature: float = 0.7):
        """
        Generate response from model given a prompt.
        
        Args:
            prompt: The formatted prompt string
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Pass temperature if supported by underlying generate implementation
        # Note: mlx_lm.generate does not support temperature parameter
        output = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Extract clean response
        return self.chat_formatter.extract_response(output)

    def format_chat(self, system: str, user: str, history: str = None) -> str:
        """
        Format a chat conversation for model generation.
        
        Args:
            system: System message/instructions
            user: Current user message
            history: Optional conversation history
            
        Returns:
            Formatted prompt ready for generation
        """
        return self.chat_formatter.build_prompt(system, user, history)

    def format_messages(self, messages: list) -> str:
        """
        Format a list of Message objects into a prompt.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Formatted prompt string
        """
        return self.chat_formatter.format_prompt(messages)

    def embed(self, texts):
        return self.embedder.encode(texts)
