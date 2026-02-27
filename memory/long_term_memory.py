"""
Long-term structured memory storage.

Design goals:
- Structured JSON storage (profile/preferences/goals/entries)
- Importance scoring for stored entries
- Explicit writes only (no background writer)
- Corruption-safe loading with backup recovery
"""

from __future__ import annotations

import json
import copy
import os
import threading
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logger import logger


class LongTermMemory:
    """Persistent long-term memory store with structured sections."""

    def __init__(self, filepath: str, max_entries: int = 500):
        self.filepath = filepath
        self.max_entries = max(50, int(max_entries))
        self._lock = threading.RLock()
        self.data = self._load() or self._default_data()

    def _default_data(self) -> Dict[str, Any]:
        return {
            "user": {},
            "preferences": {},
            "goals": [],
            "system_state": {},
            "entries": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _backup_corrupt(self, raw_text: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.filepath}.corrupt.{ts}"
        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            logger.warning(f"Backed up corrupt long-term memory to {backup_path}")
        except Exception as exc:
            logger.error(f"Failed to back up corrupt long-term memory: {exc}")

    def _load(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not os.path.exists(self.filepath):
                return None
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    raw = f.read()
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(f"Long-term memory JSON corrupted: {self.filepath}")
                try:
                    self._backup_corrupt(raw)
                except Exception:
                    pass
                return None
            except Exception as exc:
                logger.error(f"Failed to load long-term memory: {exc}")
                return None

            if not isinstance(payload, dict):
                logger.warning("Long-term memory has invalid format; reinitializing")
                return None

            payload.setdefault("user", {})
            payload.setdefault("preferences", {})
            payload.setdefault("goals", [])
            payload.setdefault("system_state", {})
            payload.setdefault("entries", [])
            payload.setdefault("created_at", datetime.now().isoformat())
            payload.setdefault("updated_at", datetime.now().isoformat())
            return payload

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self.filepath)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def _compute_importance(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        score = 0.3
        text_l = (text or "").lower()
        important_cues = (
            "my name is",
            "i prefer",
            "my goal",
            "important",
            "remember",
            "deadline",
            "born in",
            "i struggle",
        )
        if any(c in text_l for c in important_cues):
            score += 0.35
        if metadata and metadata.get("source") == "profile_extraction":
            score += 0.2
        if len(text) > 120:
            score += 0.05
        return min(1.0, max(0.0, score))

    def add_entry(self, text: str, metadata: Optional[Dict[str, Any]] = None, importance: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            now = datetime.now().isoformat()
            score = importance if importance is not None else self._compute_importance(text, metadata)
            entry = {
                "id": len(self.data["entries"]),
                "text": text,
                "metadata": metadata or {},
                "importance": float(score),
                "created_at": now,
                "updated_at": now,
            }
            self.data["entries"].append(entry)

            # Enforce memory cap (prevent overflow).
            if len(self.data["entries"]) > self.max_entries:
                self.data["entries"] = self.data["entries"][-self.max_entries:]
                for idx, item in enumerate(self.data["entries"]):
                    item["id"] = idx

            self.data["updated_at"] = now
            return entry

    def top_entries(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            entries = self.data.get("entries", [])
            sorted_entries = sorted(entries, key=lambda e: e.get("importance", 0.0), reverse=True)
            return sorted_entries[: max(1, int(limit))]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            keys = key.split(".")
            current = self.data
            for segment in keys[:-1]:
                if segment not in current or not isinstance(current[segment], dict):
                    current[segment] = {}
                current = current[segment]
            current[keys[-1]] = value
            self.data["updated_at"] = datetime.now().isoformat()

    def get(self, key: str) -> Any:
        with self._lock:
            keys = key.split(".")
            current: Any = self.data
            for segment in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(segment)
            return current

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data)

    def save(self) -> None:
        with self._lock:
            self._ensure_parent_dir()
            parent = os.path.dirname(self.filepath) or "."
            fd, temp_file = tempfile.mkstemp(prefix="long_term_", suffix=".tmp", dir=parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, self.filepath)
            except Exception as exc:
                logger.error(f"Failed to save long-term memory: {exc}")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
