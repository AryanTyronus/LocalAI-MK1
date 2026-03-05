"""
FAISS + SQLite vector store for retrieval chunks.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional

import faiss
import numpy as np

from core.config import BASE_DIR
from core.logger import logger


class KnowledgeVectorStore:
    def __init__(self, store_dir: Optional[str] = None):
        self._store_dir = store_dir or os.path.join(BASE_DIR, "memory", "knowledge_index")
        os.makedirs(self._store_dir, exist_ok=True)
        self._db_path = os.path.join(self._store_dir, "knowledge.sqlite3")
        self._index_path = os.path.join(self._store_dir, "knowledge.faiss")
        self._emb_path = os.path.join(self._store_dir, "embeddings.npy")

        self.index: Optional[faiss.IndexFlatL2] = None
        self._embeddings: Optional[np.ndarray] = None
        self._rows: List[Dict] = []
        self._dim: Optional[int] = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    row_idx INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_type TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            conn.commit()

    def build(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        if embeddings.size == 0 or len(chunks) == 0:
            self.index = None
            self._embeddings = None
            self._rows = []
            self._dim = None
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        self._dim = int(vectors.shape[1])
        self.index = faiss.IndexFlatL2(self._dim)
        self.index.add(vectors)
        self._embeddings = vectors
        self._rows = list(chunks)

        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            payload = [
                (
                    i,
                    row["doc_id"],
                    row["doc_name"],
                    row.get("path", ""),
                    int(row.get("page_number", 1)),
                    int(row.get("chunk_index", 0)),
                    row.get("text", ""),
                    row.get("source_type", "unknown"),
                )
                for i, row in enumerate(self._rows)
            ]
            conn.executemany(
                """
                INSERT INTO chunks (
                    row_idx, doc_id, doc_name, path, page_number, chunk_index, text, source_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()

        faiss.write_index(self.index, self._index_path)
        np.save(self._emb_path, vectors)
        logger.info(f"KnowledgeVectorStore built with {len(self._rows)} chunks (dim={self._dim})")

    def _load_rows(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT row_idx, doc_id, doc_name, path, page_number, chunk_index, text, source_type
                FROM chunks
                ORDER BY row_idx ASC
                """
            ).fetchall()
        return [
            {
                "id": int(r[0]),
                "doc_id": r[1],
                "doc_name": r[2],
                "path": r[3],
                "page_number": int(r[4]),
                "chunk_index": int(r[5]),
                "text": r[6],
                "source_type": r[7],
            }
            for r in rows
        ]

    def load(self) -> bool:
        if not (os.path.exists(self._index_path) and os.path.exists(self._db_path) and os.path.exists(self._emb_path)):
            return False
        try:
            self.index = faiss.read_index(self._index_path)
            self._embeddings = np.load(self._emb_path).astype(np.float32)
            self._rows = self._load_rows()
            self._dim = int(self._embeddings.shape[1]) if self._embeddings.ndim == 2 else None
            return self.index is not None and bool(self._rows)
        except Exception as exc:
            logger.warning(f"Failed to load persisted knowledge index: {exc}")
            self.index = None
            self._embeddings = None
            self._rows = []
            self._dim = None
            return False

    def search(self, query_vec: np.ndarray, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict]:
        if self.index is None or self._embeddings is None or not self._rows:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(max(1, int(top_k)), len(self._rows))

        if doc_id:
            candidate_idxs = [i for i, row in enumerate(self._rows) if row["doc_id"] == doc_id]
            if not candidate_idxs:
                return []
            subset = self._embeddings[candidate_idxs]
            dists = np.linalg.norm(subset - q[0], axis=1)
            order = np.argsort(dists)[: min(k, len(candidate_idxs))]
            return [
                {"chunk": self._rows[candidate_idxs[i]], "score": float(dists[i])}
                for i in order
            ]

        distances, indices = self.index.search(q, k=k)
        results: List[Dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._rows):
                continue
            results.append({"chunk": self._rows[int(idx)], "score": float(dist)})
        return results

    def list_chunks(self) -> List[Dict]:
        return list(self._rows)

