"""
ModelManager - Handles LLM model loading and generation.
Supports streaming generation stub for future implementation.
"""

import os
from typing import Generator
import numpy as np
from sentence_transformers import SentenceTransformer
from mlx_lm import generate as mlx_generate
from core.config import MODEL_NAME
from core.logger import logger
from core.chat_formatter import ChatFormatter, Message, Role, ModelType


class FakeModelManager:
    """Lightweight fake model for development mode."""
    
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

    def generate_stream(self, prompt, max_tokens=300, temperature: float = 0.7) -> Generator[str]:
        """
        Streaming generation stub for FakeModelManager.
        
        Yields response chunks (currently yields full response).
        Future implementation will stream actual tokens.
        """
        response = self.generate(prompt, max_tokens, temperature)
        # Simulate streaming by yielding small chunks
        # Split into words and yield incrementally
        words = response.split()
        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = ' '.join(words[i:i+3])
            if chunk:
                yield chunk + ' '
        # Ensure we yield the complete response
        if not words:
            yield response

    def format_chat(self, system: str, user: str, history: str = None) -> str:
        return (system or '') + "\n\n" + (user or '')

    def format_messages(self, messages: list) -> str:
        return '\n'.join([f"{m.role}: {m.content}" for m in messages])

    def embed(self, texts):
        return np.zeros((len(texts), self._embed_dim), dtype=float)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using approximation: len(text.split()) * 1.3
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return int(len(text.split()) * 1.3)


class ModelManager:
    """Real model manager for production mode."""
    
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

    def generate_stream(self, prompt, max_tokens=300, temperature: float = 0.7) -> Generator[str]:
        """
        Streaming generation for ModelManager.
        
        Since mlx_lm may not support true streaming, we generate the full response
        and yield it in small chunks to simulate streaming.
        
        Args:
            prompt: The formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Yields:
            Response chunks
        """
        # Generate full response first
        response = self.generate(prompt, max_tokens, temperature)
        
        # Simulate streaming by yielding small chunks
        # Split into words and yield incrementally
        words = response.split()
        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = ' '.join(words[i:i+3])
            if chunk:
                yield chunk + ' '
        # Ensure we yield the complete response
        if not words:
            yield response
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using approximation: len(text.split()) * 1.3
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return int(len(text.split()) * 1.3)

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

