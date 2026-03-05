"""
Memory retriever for SQLite-backed long-term memory.
"""

from __future__ import annotations

from typing import List

from memory.memory_store import MemoryStore


class MemoryRetriever:
    def __init__(self, store: MemoryStore):
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        rows = self._store.retrieve_long_term(query=query, limit=top_k)
        return [r.get("text", "") for r in rows if (r.get("text") or "").strip()]

