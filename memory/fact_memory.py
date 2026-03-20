"""Durable fact extraction, deduplication, and consolidation."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.logger import logger


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    timestamp: str
    source: str = "conversation"

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}".strip()


class FactMemory:
    def __init__(self, model_manager, filepath: str, similarity_threshold: float = 0.88):
        self._model_manager = model_manager
        self._filepath = filepath
        self._similarity_threshold = similarity_threshold
        self._lock = threading.RLock()
        self._facts: List[Dict] = []
        self._dirty = False
        self._last_write_ts = 0.0
        self._debounce_seconds = 1.5
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._filepath):
            self._facts = []
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._facts = data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"FactMemory load failed: {exc}")
            self._facts = []

    def save(self, force: bool = False) -> None:
        with self._lock:
            if not self._dirty and not force:
                return
            now = time.time()
            if not force and (now - self._last_write_ts) < self._debounce_seconds:
                return
            os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, indent=2)
            self._dirty = False
            self._last_write_ts = now

    def all_facts(self) -> List[Dict]:
        with self._lock:
            return list(self._facts)

    def extract(self, text: str) -> List[ExtractedFact]:
        if not text:
            return []
        raw = text.strip()
        now = datetime.now().astimezone().isoformat()
        out: List[ExtractedFact] = []

        # Priority extraction patterns for durable profile facts.
        patterns: List[Tuple[re.Pattern, str, str]] = [
            (re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\s'\-]{1,60})", re.IGNORECASE), "user", "name_is"),
            (re.compile(r"\bi(?:'m| am)\s+([A-Za-z][A-Za-z\s'\-]{1,40})\b", re.IGNORECASE), "user", "is"),
            (re.compile(r"\bi like\s+([^.!?]{2,80})", re.IGNORECASE), "user", "likes"),
            (re.compile(r"\bi prefer\s+([^.!?]{2,80})", re.IGNORECASE), "user", "prefers"),
            (re.compile(r"\bmy goal is to\s+([^.!?]{2,100})", re.IGNORECASE), "user", "goal_is"),
            (re.compile(r"\bi struggle with\s+([^.!?]{2,80})", re.IGNORECASE), "user", "struggles_with"),
        ]

        for pattern, subject, predicate in patterns:
            for match in pattern.finditer(raw):
                obj = re.split(r"\b(and|but|because|so)\b", match.group(1), maxsplit=1)[0].strip(" .,!?")
                if len(obj) < 2:
                    continue
                out.append(ExtractedFact(subject=subject, predicate=predicate, object=obj, timestamp=now))

        return out

    def _embed(self, text: str) -> List[float]:
        try:
            vec = self._model_manager.embed([text])[0]
            return [float(x) for x in vec]
        except Exception:
            return []

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def add_facts(self, facts: List[ExtractedFact]) -> int:
        added = 0
        with self._lock:
            for fact in facts:
                text = fact.text
                embedding = self._embed(text)
                is_dup = False
                for existing in self._facts[-300:]:
                    if existing.get("subject") == fact.subject and existing.get("predicate") == fact.predicate:
                        if str(existing.get("object", "")).strip().lower() == fact.object.lower():
                            is_dup = True
                            break
                    sim = self._cosine(embedding, existing.get("embedding", [])) if embedding else 0.0
                    if sim >= self._similarity_threshold:
                        is_dup = True
                        break
                if is_dup:
                    continue
                self._facts.append({
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "timestamp": fact.timestamp,
                    "source": fact.source,
                    "embedding": embedding,
                })
                added += 1
            if added > 0:
                self._dirty = True
        if added > 0:
            self.save()
        return added

    def top_relevant(self, query: str, limit: int = 5, token_budget: int = 220) -> List[str]:
        q_embed = self._embed(query)
        scored = []
        with self._lock:
            for row in self._facts:
                text = f"{row.get('subject')} {row.get('predicate')} {row.get('object')}"
                score = self._cosine(q_embed, row.get("embedding", [])) if q_embed else 0.0
                if score <= 0:
                    if any(tok in text.lower() for tok in query.lower().split()[:4]):
                        score = 0.2
                scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = []
        used_tokens = 0
        for score, text in scored[: max(limit * 4, limit)]:
            if len(selected) >= limit:
                break
            est = max(1, int(len(text) / 4))
            if used_tokens + est > token_budget:
                continue
            selected.append(text)
            used_tokens += est
        return selected

    def consolidate(self, stale_days: int = 120) -> int:
        """Merge duplicates and trim stale low-value facts."""
        now = datetime.now().astimezone()
        with self._lock:
            merged: Dict[Tuple[str, str, str], Dict] = {}
            removed = 0
            for row in self._facts:
                key = (
                    str(row.get("subject", "")).strip().lower(),
                    str(row.get("predicate", "")).strip().lower(),
                    str(row.get("object", "")).strip().lower(),
                )
                if key in merged:
                    removed += 1
                    continue
                ts = row.get("timestamp", "")
                try:
                    age_days = (now - datetime.fromisoformat(ts)).days
                except Exception:
                    age_days = 0
                if age_days > stale_days and key[1] not in {"name_is", "goal_is", "prefers"}:
                    removed += 1
                    continue
                merged[key] = row
            self._facts = list(merged.values())
            if removed > 0:
                self._dirty = True
        if removed > 0:
            self.save(force=True)
        return removed
