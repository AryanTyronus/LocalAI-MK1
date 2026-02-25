"""
AIService - Pure orchestrator for AI generation.

This service is now a thin wrapper that:
- Accepts GenerationPipeline via constructor
- Does NOT instantiate MemoryManager, ModelManager, etc internally
- Does NOT import Flask or request objects
- Does NOT access global state
- Only exposes:
  - generate_response(user_message: str, mode: str = "chat") -> str
  - generate_stream(user_message: str, mode: str = "chat") -> Generator[str]

All heavy logic is in GenerationPipeline.
"""

from typing import Generator

from core.config import Config
from core.generation_pipeline import GenerationPipeline
from core.logger import logger


class AIService:
    """
    AIService - Pure orchestrator for AI generation.
    
    This class is now a thin wrapper that delegates all generation logic
    to GenerationPipeline. It does not contain any heavy business logic.
    """
    
    def __init__(
        self,
        pipeline: GenerationPipeline,
        config: Config = None
    ):
        """
        Initialize AIService with dependencies.
        
        Args:
            pipeline: GenerationPipeline instance for all generation logic
            config: Config instance (optional, uses singleton if not provided)
        """
        self._pipeline = pipeline
        self._config = config or Config()
        
        logger.info("AIService initialized as pure orchestrator")
    
    def generate_response(self, user_message: str, mode: str = "chat") -> str:
        """
        Generate a response for the user message.
        
        This is the main entry point for non-streaming generation.
        
        Args:
            user_message: The user's input message
            mode: Operation mode (chat, coding, research, agent)
            
        Returns:
            Generated response string
        """
        return self._pipeline.generate(user_message, mode)
    
    def generate_stream(self, user_message: str, mode: str = "chat") -> Generator[str]:
        """
        Generate a streaming response for the user message.
        
        This is the main entry point for streaming generation.
        Yields content chunks as they arrive, then yields final token info.
        
        Args:
            user_message: The user's input message
            mode: Operation mode (chat, coding, research, agent)
            
        Yields:
            Content chunks (str), then JSON-encoded final token info
        """
        for item in self._pipeline.run_stream(user_message, mode):
            if 'content' in item:
                yield item['content']
            elif 'done' in item:
                # Yield token counts as special final message
                import json
                yield f"__TOKENS__:{json.dumps(item)}"
    
    # ===================
    # Legacy compatibility methods
    # ===================
    
    def ask(self, user_input: str, mode: str = "chat") -> str:
        """
        Legacy compatibility method.
        
        Maps to generate_response for backward compatibility.
        
        Args:
            user_input: The user's input message
            mode: Operation mode
            
        Returns:
            Generated response string
        """
        return self.generate_response(user_input, mode)

