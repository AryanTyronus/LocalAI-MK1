"""
Project-namespaced vector memory with lazy index lifecycle.

Design goals:
- Namespaced storage per project
- Lazy loading of embeddings/index
- Top-k retrieval only
- No background indexing or workers
"""

from __future__ import annotations

import json
import os
import threading
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logger import logger

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    faiss = None


class VectorMemory:
    """Lazy, namespaced vector memory."""

    def __init__(
        self,
        model_manager,
        base_dir: str,
        namespace: str,
        max_entries: int = 2000,
    ):
        self._model_manager = model_manager
        self._namespace = self._sanitize_namespace(namespace)
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._loaded = False
        self._index = None
        self._embedding_dim = None
        self._memories: List[Dict[str, Any]] = []

        ns_dir = os.path.join(base_dir, "memory", "vector", self._namespace)
        self._json_path = os.path.join(ns_dir, "memories.json")
        self._index_path = os.path.join(ns_dir, "index.faiss")

    def _sanitize_namespace(self, namespace: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (namespace or "default"))
        return clean[:80] or "default"

    def _ensure_dir(self) -> None:
        parent = os.path.dirname(self._json_path)
        os.makedirs(parent, exist_ok=True)

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True

            if os.path.exists(self._json_path):
                try:
                    with open(self._json_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        self._memories = payload
                except json.JSONDecodeError:
                    logger.error(f"Vector memory JSON corrupted: {self._json_path}")
                    try:
                        with open(self._json_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup = f"{self._json_path}.corrupt.{ts}"
                        with open(backup, "w", encoding="utf-8") as out:
                            out.write(raw)
                        logger.warning(f"Backed up corrupt vector memory to {backup}")
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error(f"Failed to load vector memory: {exc}")

            if self._memories:
                emb0 = self._memories[0].get("embedding", [])
                self._embedding_dim = len(emb0) if isinstance(emb0, list) else None

            if faiss is not None and os.path.exists(self._index_path):
                try:
                    self._index = faiss.read_index(self._index_path)
                except Exception as exc:
                    logger.warning(f"Failed to load FAISS index, rebuilding lazily: {exc}")
                    self._index = None

    def _build_index(self) -> None:
        if faiss is None or not self._memories:
            self._index = None
            return
        try:
            vectors = np.array([m["embedding"] for m in self._memories], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] == 0:
                self._index = None
                return
            self._embedding_dim = vectors.shape[1]
            self._index = faiss.IndexFlatL2(self._embedding_dim)
            self._index.add(vectors)
        except Exception as exc:
            logger.error(f"Failed to build vector index: {exc}")
            self._index = None

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> bool:
        with self._lock:
            self._load()
            try:
                embedding = self._model_manager.embed([text])[0]
                embedding = np.array(embedding, dtype=np.float32)
            except Exception as exc:
                logger.error(f"Embedding generation failed: {exc}")
                return False

            now = datetime.now().isoformat()
            entry = {
                "id": len(self._memories),
                "text": text,
                "embedding": embedding.tolist(),
                "metadata": metadata or {},
                "importance": float(max(0.0, min(1.0, importance))),
                "created_at": now,
                "last_accessed": now,
                "access_count": 1,
            }
            self._memories.append(entry)

            if len(self._memories) > self._max_entries:
                self._memories = self._memories[-self._max_entries:]
                for idx, item in enumerate(self._memories):
                    item["id"] = idx
                self._index = None  # force rebuild, prevent stale ids

            # Incremental add when possible, fallback to rebuild.
            if faiss is not None:
                if self._index is None:
                    self._build_index()
                else:
                    try:
                        self._index.add(np.array([embedding], dtype=np.float32))
                    except Exception:
                        self._build_index()
            return True

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        with self._lock:
            self._load()
            if not self._memories:
                return []

            k = max(1, min(int(top_k), len(self._memories)))
            try:
                query_vec = np.array(self._model_manager.embed([query])[0], dtype=np.float32)
            except Exception as exc:
                logger.error(f"Vector search embedding failed: {exc}")
                return []

            # FAISS path
            if faiss is not None:
                if self._index is None:
                    self._build_index()
                if self._index is not None:
                    try:
                        distances, indices = self._index.search(np.array([query_vec], dtype=np.float32), k=k)
                        return self._collect_results(indices[0], distances[0])
                    except Exception as exc:
                        logger.warning(f"FAISS search failed; falling back to linear scan: {exc}")

            # Fallback: linear search
            return self._linear_search(query_vec, k)

    def _collect_results(self, indices, distances) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        now = datetime.now().isoformat()
        for idx_val, dist in zip(indices, distances):
            if idx_val < 0 or idx_val >= len(self._memories):
                continue
            memory = self._memories[int(idx_val)]
            memory["access_count"] = int(memory.get("access_count", 0)) + 1
            memory["last_accessed"] = now
            similarity = float(1.0 / (1.0 + dist))
            results.append((memory.get("text", ""), similarity))
        return results

    def _linear_search(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        qn = np.linalg.norm(query_vec) + 1e-8
        now = datetime.now().isoformat()
        for memory in self._memories:
            emb = np.array(memory.get("embedding", []), dtype=np.float32)
            if emb.size == 0 or emb.shape != query_vec.shape:
                continue
            sim = float(np.dot(query_vec, emb) / (qn * (np.linalg.norm(emb) + 1e-8)))
            scored.append((memory.get("text", ""), sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        # update access metrics for returned entries
        top = scored[:k]
        top_texts = {t for t, _ in top}
        for memory in self._memories:
            if memory.get("text", "") in top_texts:
                memory["access_count"] = int(memory.get("access_count", 0)) + 1
                memory["last_accessed"] = now
        return top

    def save(self) -> None:
        with self._lock:
            self._load()
            self._ensure_dir()
            try:
                parent = os.path.dirname(self._json_path) or "."
                fd, tmp = tempfile.mkstemp(prefix="vector_mem_", suffix=".tmp", dir=parent)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._memories, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self._json_path)
            except Exception as exc:
                logger.error(f"Failed to save vector memory JSON: {exc}")

            if faiss is not None and self._index is not None:
                try:
                    faiss.write_index(self._index, self._index_path)
                except Exception as exc:
                    logger.error(f"Failed to save FAISS index: {exc}")

    def get_info(self) -> Dict[str, Any]:
        with self._lock:
            self._load()
            return {
                "namespace": self._namespace,
                "count": len(self._memories),
                "loaded": self._loaded,
                "has_index": self._index is not None,
            }

    def export_memories(self) -> List[Dict[str, Any]]:
        """Return a shallow-copy snapshot for compatibility exports."""
        with self._lock:
            self._load()
            return [dict(m) for m in self._memories]
