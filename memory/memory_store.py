"""
SQLite-backed memory store.

Stores:
- short-term conversation records
- long-term structured facts
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from core.config import BASE_DIR
from core.logger import logger


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(BASE_DIR, "memory", "memory.sqlite3")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_term_ts ON short_term_messages(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_term_category ON long_term_facts(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_term_ts ON long_term_facts(timestamp)"
            )
            conn.commit()

    # --------------------
    # Short-term memory
    # --------------------
    def add_short_term(self, role: str, content: str, timestamp: Optional[str] = None) -> None:
        ts = timestamp or datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO short_term_messages(role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, ts),
            )
            conn.commit()

    def get_recent_short_term(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM short_term_messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        rows.reverse()
        return [
            {"id": int(r[0]), "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in rows
        ]

    # --------------------
    # Long-term memory
    # --------------------
    def add_long_term_fact(
        self,
        text: str,
        category: str = "general",
        source: str = "memory_manager",
        importance: float = 0.5,
        timestamp: Optional[str] = None,
    ) -> None:
        if not (text or "").strip():
            return
        ts = timestamp or datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_facts(text, category, source, importance, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text.strip(), category, source, float(max(0.0, min(1.0, importance))), ts),
            )
            conn.commit()

    def retrieve_long_term(self, query: str, limit: int = 5) -> List[Dict]:
        text = (query or "").strip()
        if not text:
            return []
        terms = [t for t in text.lower().split() if len(t) >= 3][:6]
        if not terms:
            return []

        with self._connect() as conn:
            all_rows = conn.execute(
                """
                SELECT id, text, category, source, importance, timestamp
                FROM long_term_facts
                ORDER BY importance DESC, id DESC
                LIMIT 300
                """
            ).fetchall()

        scored = []
        for row in all_rows:
            body = (row[1] or "").lower()
            score = float(row[4])
            score += sum(0.25 for term in terms if term in body)
            if score <= float(row[4]):
                continue
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, int(limit))]
        return [
            {
                "id": int(r[0]),
                "text": r[1],
                "category": r[2],
                "source": r[3],
                "importance": float(r[4]),
                "timestamp": r[5],
                "score": float(score),
            }
            for score, r in top
        ]

    # --------------------
    # Legacy compatibility
    # --------------------
    def add_memory(self, text: str) -> None:
        self.add_long_term_fact(text=text, category="legacy", source="legacy_memory_store", importance=0.4)

    def search(self, query: str) -> str:
        results = self.retrieve_long_term(query, limit=3)
        return "\n".join(item["text"] for item in results)

