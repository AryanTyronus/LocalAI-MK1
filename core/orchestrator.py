"""
Application orchestration layer.

Coordinates dependency initialization and optional model warm-up.
"""

from __future__ import annotations

from core.dependency_container import DependencyContainer
from core.logger import logger


class AppOrchestrator:
    """Owns startup lifecycle and service access."""

    def __init__(self):
        self._container = None
        self._ai_service = None

    def get_ai_service(self):
        """Lazily initialize and return AIService."""
        if self._ai_service is None:
            self._container = DependencyContainer()
            self._ai_service = self._container.get_ai_service()
            logger.info("AIService initialized successfully")
        return self._ai_service

    def warm_model(self) -> None:
        """Load model into memory and keep it idle."""
        if self._container is None:
            self._container = DependencyContainer()
        model_manager = self._container.get_model_manager()
        if hasattr(model_manager, "warm_up"):
            model_manager.warm_up()
        logger.info("Model warm-up completed")

