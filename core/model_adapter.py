"""
Model Adapter Layer - Abstract interface for model generation.

This module provides:
- BaseModelInterface: Abstract interface for all model adapters
- LocalModelAdapter: Adapter for current MLX model
- FutureRemoteModelAdapter: Stub for future remote/cloud models

All generation must go through the interface - no direct model calls in pipeline.
This future-proofs scaling to different model backends.
"""

from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Dict, Any
from dataclasses import dataclass

from core.logger import logger
from core.config import Config


@dataclass
class GenerationResult:
    """Result from model generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration: float
    model_name: str


@dataclass
class GenerationStreamResult:
    """Result from streaming model generation."""
    chunks: List[str]
    final_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration: float
    model_name: str


class BaseModelInterface(ABC):
    """
    Abstract interface for model adapters.
    
    All model implementations must implement this interface.
    This enables:
    - Easy switching between local and remote models
    - Standardized generation API
    - Future-proof scaling
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> GenerationResult:
        """
        Generate a response from the model.
        
        Args:
            prompt: Formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            GenerationResult with text and metrics
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Stream generation from the model.
        
        Args:
            prompt: Formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Yields:
            Response chunks
        """
        pass
    
    @abstractmethod
    def format_chat(
        self, 
        system: str, 
        user: str, 
        history: Optional[str] = None
    ) -> str:
        """
        Format a chat conversation for generation.
        
        Args:
            system: System message
            user: User message
            history: Optional conversation history
            
        Returns:
            Formatted prompt
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available.
        
        Returns:
            True if model can be used
        """
        pass


class LocalModelAdapter(BaseModelInterface):
    """
    Adapter for local MLX model.
    
    Wraps the existing ModelManager for backward compatibility.
    """
    
    def __init__(self, model_manager):
        """
        Initialize with ModelManager instance.
        
        Args:
            model_manager: ModelManager instance
        """
        self._model_manager = model_manager
        self._model_name = "local-mlx"
        
        # Try to get actual model name from manager
        if hasattr(model_manager, 'model') and hasattr(model_manager.model, 'model_id'):
            try:
                self._model_name = model_manager.model.model_id
            except Exception:
                pass
        
        logger.info(f"LocalModelAdapter initialized with model: {self._model_name}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> GenerationResult:
        """Generate using local MLX model."""
        import time
        start = time.time()
        
        text = self._model_manager.generate(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        duration = time.time() - start
        
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(text)
        
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration=duration,
            model_name=self._model_name
        )
    
    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Stream using local MLX model."""
        # Delegate to model's streaming method
        for chunk in self._model_manager.generate_stream(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature
        ):
            yield chunk
    
    def format_chat(
        self, 
        system: str, 
        user: str, 
        history: Optional[str] = None
    ) -> str:
        """Format chat using model manager."""
        return self._model_manager.format_chat(system, user, history)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using model manager."""
        return self._model_manager.estimate_tokens(text)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using model manager."""
        return self._model_manager.embed(texts)
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return self._model_manager is not None


class FutureRemoteModelAdapter(BaseModelInterface):
    """
    Stub adapter for future remote/cloud models.
    
    This adapter can be enabled when the system needs to scale
    to cloud-based LLMs (OpenAI, Anthropic, etc.).
    
    Currently returns error responses - implementation TBD.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize remote adapter stub.
        
        Args:
            config: Config instance for settings
        """
        self._config = config or Config()
        self._model_name = "remote-stub"
        self._api_endpoint = self._config.get('model', {}).get('remote_endpoint', '')
        self._api_key = self._config.get('model', {}).get('remote_api_key', '')
        
        logger.warning("FutureRemoteModelAdapter initialized - STUB IMPLEMENTATION")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> GenerationResult:
        """Stub - not implemented yet."""
        logger.error("Remote model adapter not implemented")
        return GenerationResult(
            text="Remote model not available - please use local model",
            prompt_tokens=self.estimate_tokens(prompt),
            completion_tokens=0,
            total_tokens=self.estimate_tokens(prompt),
            duration=0.0,
            model_name=self._model_name
        )
    
    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Stub - not implemented yet."""
        logger.error("Remote model streaming not implemented")
        yield "Remote model not available - please use local model"
    
    def format_chat(
        self, 
        system: str, 
        user: str, 
        history: Optional[str] = None
    ) -> str:
        """Format chat for remote API."""
        # Placeholder - would format for specific API
        return f"System: {system}\n\nUser: {user}"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens - same as local."""
        return max(1, int(len(text.split()) * 1.3))
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Stub - would call remote embedding API."""
        logger.error("Remote embeddings not implemented")
        # Return zero embeddings to avoid crashes
        return [[0.0] * 384 for _ in texts]
    
    def is_available(self) -> bool:
        """Remote adapter not available (stub)."""
        return False


class ModelAdapterFactory:
    """
    Factory for creating model adapters.
    
    Provides easy switching between local and remote models
    via configuration.
    """
    
    _adapters: Dict[str, BaseModelInterface] = {}
    _current: Optional[BaseModelInterface] = None
    
    @classmethod
    def create_adapter(
        cls, 
        adapter_type: str = "local",
        model_manager=None,
        config: Config = None
    ) -> BaseModelInterface:
        """
        Create a model adapter.
        
        Args:
            adapter_type: "local" or "remote"
            model_manager: ModelManager instance (for local)
            config: Config instance
            
        Returns:
            BaseModelInterface implementation
        """
        if adapter_type == "local":
            if "local" not in cls._adapters:
                cls._adapters["local"] = LocalModelAdapter(model_manager)
            cls._current = cls._adapters["local"]
            return cls._current
        
        elif adapter_type == "remote":
            if "remote" not in cls._adapters:
                cls._adapters["remote"] = FutureRemoteModelAdapter(config)
            cls._current = cls._adapters["remote"]
            return cls._current
        
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    @classmethod
    def get_current_adapter(cls) -> Optional[BaseModelInterface]:
        """Get the current active adapter."""
        return cls._current
    
    @classmethod
    def set_adapter(cls, adapter: BaseModelInterface) -> None:
        """Set the current adapter."""
        cls._current = adapter
        logger.info(f"Model adapter set to: {adapter.model_name}")

