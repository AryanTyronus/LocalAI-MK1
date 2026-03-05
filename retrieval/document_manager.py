"""
DocumentManager backed by local RAG components.

Public interface is preserved for the existing pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import faiss
import numpy as np

from core.logger import logger
from core.model_manager import ModelManager
from knowledge.retriever import KnowledgeRetriever


class DocumentManager:
    def __init__(
        self,
        chunk_tokens: int = 800,
        overlap: int = 100,
        index_file: Optional[str] = None,
        lazy_init: bool = True,
    ):
        self.model_manager = ModelManager.get_instance()
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.index_file = index_file
        self._lazy_init = lazy_init
        self._loaded = False

        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
        self.active_doc_id: Optional[str] = None

        # Legacy/manual index path (kept for compatibility with tests/utilities).
        self.index: Optional[faiss.IndexFlatL2] = None
        self.embedding_dim: Optional[int] = None

        # Upgrade-1 RAG path.
        self._retriever = KnowledgeRetriever()

        if not self._lazy_init:
            self._load_and_index()

    def _load_and_index(self) -> None:
        if self._loaded:
            return
        self._retriever.ensure_ready()
        self.documents = self._retriever.list_documents()
        self.chunks = self._retriever.list_chunks()
        self._loaded = True
        logger.info(f"DocumentManager ready with {len(self.chunks)} retrieval chunks")

    def _build_index(self) -> None:
        """
        Build an in-memory FAISS index from current self.chunks.
        This is the compatibility path used by older scripts/tests.
        """
        if not self.chunks:
            self.index = None
            self.embedding_dim = None
            return

        texts = [c.get("text", "") for c in self.chunks]
        vectors = np.asarray(self.model_manager.embed(texts), dtype=np.float32)
        if vectors.size == 0:
            self.index = None
            self.embedding_dim = None
            return
        self.embedding_dim = int(vectors.shape[1])
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(vectors)
        logger.info(f"Built manual document index with {len(self.chunks)} chunks (dim={self.embedding_dim})")

    def _ensure_index(self) -> None:
        if self._lazy_init and not self._loaded:
            self._load_and_index()
        if self.index is None and self.chunks and not self._loaded:
            # Fallback if external code injected chunks before loading.
            self._build_index()

    def _search_manual_index(self, query: str, top_k: int, doc_id: Optional[str]) -> List[Dict]:
        if self.index is None or not self.chunks:
            return []
        q_vec = np.asarray(self.model_manager.embed([query]), dtype=np.float32)
        if q_vec.size == 0:
            return []

        if doc_id:
            scoped = [i for i, c in enumerate(self.chunks) if c.get("doc_id") == doc_id]
            if not scoped:
                return []
            embs = np.asarray([self.model_manager.embed([self.chunks[i].get("text", "")])[0] for i in scoped], dtype=np.float32)
            idx = faiss.IndexFlatL2(int(embs.shape[1]))
            idx.add(embs)
            k = min(max(1, int(top_k)), embs.shape[0])
            distances, matched = idx.search(q_vec, k=k)
            return [
                {"chunk": self.chunks[scoped[int(i)]], "score": float(d)}
                for d, i in zip(distances[0], matched[0])
            ]

        k = min(max(1, int(top_k)), len(self.chunks))
        distances, matched = self.index.search(q_vec, k=k)
        return [
            {"chunk": self.chunks[int(i)], "score": float(d)}
            for d, i in zip(distances[0], matched[0])
            if 0 <= int(i) < len(self.chunks)
        ]

    def search(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict]:
        self._ensure_index()
        if not (query or "").strip():
            return []

        # If a manual index exists (compat mode), use it.
        if self.index is not None:
            return self._search_manual_index(query=query, top_k=top_k, doc_id=doc_id)

        # Default path: new retriever backend.
        return self._retriever.search(query=query, top_k=top_k, doc_id=doc_id or self.active_doc_id)

    def set_active_document(self, doc_id: Optional[str]) -> None:
        self.active_doc_id = doc_id

    def list_documents(self) -> List[Dict]:
        if self._lazy_init and not self._loaded:
            self._load_and_index()
        return list(self.documents)

    def get_active_document(self) -> Optional[Dict]:
        if self._lazy_init and not self._loaded:
            self._load_and_index()
        if not self.active_doc_id:
            return None
        for item in self.documents:
            if item.get("doc_id") == self.active_doc_id:
                return item
        return None

    def save_index(self) -> None:
        if self.index is None or not self.index_file:
            return
        faiss.write_index(self.index, self.index_file)

    def load_index(self) -> None:
        if not self.index_file:
            return
        try:
            self.index = faiss.read_index(self.index_file)
        except Exception:
            self.index = None

