"""
Embedding service for local RAG.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.logger import logger


class EmbeddingService:
    """
    Sentence-transformer wrapper with a practical fallback model.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None
        self._offline_fallback = False
        self._hash_dim = 384

    def _ensure_model(self) -> SentenceTransformer:
        if self._offline_fallback:
            return None
        if self._model is not None:
            return self._model
        try:
            self._model = SentenceTransformer(self._model_name, local_files_only=True)
            logger.info(f"Embedding model loaded: {self._model_name}")
            return self._model
        except Exception as exc:
            fallback = "all-MiniLM-L6-v2"
            logger.warning(
                f"Embedding model '{self._model_name}' failed ({type(exc).__name__}). "
                f"Falling back to '{fallback}'."
            )
            try:
                self._model = SentenceTransformer(fallback, local_files_only=True)
                self._model_name = fallback
                logger.info(f"Embedding model loaded: {fallback}")
                return self._model
            except Exception as fallback_exc:
                logger.warning(
                    f"Fallback embedding model '{fallback}' failed ({type(fallback_exc).__name__}). "
                    "Using deterministic hash embeddings."
                )
                self._offline_fallback = True
                self._model = None
                return None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros((self._hash_dim,), dtype=np.float32)
        data = (text or "").encode("utf-8", errors="ignore")
        if not data:
            return vec
        for idx in range(0, len(data), 4):
            block = data[idx:idx + 4]
            digest = hashlib.md5(block).digest()
            pos = int.from_bytes(digest[:2], "little") % self._hash_dim
            vec[pos] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model = self._ensure_model()
        values = list(texts)
        if not values:
            return np.zeros((0, 0), dtype=np.float32)
        if model is None:
            return np.asarray([self._hash_embed(t) for t in values], dtype=np.float32)
        vectors = model.encode(values)
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        vectors = self.encode([text or ""])
        if vectors.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return vectors[0]
