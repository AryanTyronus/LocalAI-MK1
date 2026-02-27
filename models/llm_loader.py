"""
Centralized LLM loading and generation utilities.

Responsibilities:
- Apply safe thread limits for macOS/MLX workloads
- Lazy-load mlx_lm only when needed
- Keep model warm (loaded) but idle (no generation loop)
"""

from __future__ import annotations

import os
import platform
from typing import Any, Tuple

from core.logger import logger


class LLMLoader:
    """Loader for MLX LLM models with runtime safeguards."""

    def __init__(self, model_name: str, max_threads: int = 4, enable_thread_control: bool = True):
        self._model_name = model_name
        self._max_threads = max(1, int(max_threads))
        self._enable_thread_control = bool(enable_thread_control)

    def _set_thread_env(self, thread_count: int, override: bool = False) -> None:
        """Apply thread-related environment limits."""
        if not self._enable_thread_control:
            return
        thread_value = str(max(1, int(thread_count)))
        for env_key in (
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            if override:
                os.environ[env_key] = thread_value
            else:
                os.environ.setdefault(env_key, thread_value)

    def _configure_threading(self) -> None:
        """
        Set conservative thread caps to avoid contention on macOS.
        """
        if not self._enable_thread_control:
            return

        self._set_thread_env(self._max_threads, override=False)

        if platform.system().lower() == "darwin":
            logger.info(f"Applied macOS thread control (max_threads={self._max_threads})")

    def set_active_mode(self) -> None:
        """Raise thread limits for active generation."""
        self._set_thread_env(self._max_threads, override=True)

    def set_idle_mode(self) -> None:
        """Throttle thread limits while idle."""
        self._set_thread_env(1, override=True)

    def load_model(self) -> Tuple[Any, Any]:
        """
        Load MLX model/tokenizer lazily.
        """
        self._configure_threading()
        from mlx_lm import load  # lazy import: avoid MLX initialization at module import time

        logger.info(f"Loading model: {self._model_name}")
        model, tokenizer = load(self._model_name)
        logger.info("Model load completed")
        return model, tokenizer

    def generate(self, model: Any, tokenizer: Any, prompt: str, max_tokens: int = 300) -> str:
        """
        Generate text from loaded model.
        """
        from mlx_lm import generate as mlx_generate  # lazy import for safety

        return mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
