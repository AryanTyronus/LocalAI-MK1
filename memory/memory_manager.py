"""
3-Layer Memory Manager (Phase 2).

Layers:
1) Short-term memory (token-aware in-memory buffer)
2) Long-term structured memory (JSON with importance-scored entries)
3) Vector memory (project-namespaced, lazy FAISS lifecycle)
"""

from __future__ import annotations

import os
import re
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import deque

from core.config import (
    SHORT_TERM_CONFIG,
    SEMANTIC_CONFIG,
    DEFAULT_TOP_K,
    INCLUDE_SEMANTIC_IN_SEARCH,
    MEMORY_FILE,
    BASE_DIR,
)
from core.logger import logger
from core.model_manager import ModelManager
from memory.long_term_memory import LongTermMemory
from memory.vector_memory import VectorMemory


def _project_namespace() -> str:
    cwd = os.getcwd()
    return os.path.basename(cwd) or "default"


class MemoryManager:
    """
    Unified memory manager with reactive, explicit persistence.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.model_manager = ModelManager.get_instance()

        # Layer 1: short-term buffer
        self.short_term_max = int(SHORT_TERM_CONFIG.get("max_messages", 10))
        self.short_term_max_tokens = int(SHORT_TERM_CONFIG.get("max_tokens", 900))
        self.short_term = deque(maxlen=self.short_term_max)

        # Keep compatibility with existing callers/tests.
        self.rolling_summaries: List[Dict[str, Any]] = []
        self.summary_trigger = 10**9  # disabled rolling summary for 3-layer design

        # Layer 2: long-term structured
        long_term_path = MEMORY_FILE
        self.long_term = LongTermMemory(filepath=long_term_path, max_entries=700)
        self._maybe_migrate_legacy_structured(long_term_path)

        # Layer 3: vector memory (project namespaced, lazy)
        namespace = _project_namespace()
        self.semantic = VectorMemory(
            model_manager=self.model_manager,
            base_dir=BASE_DIR,
            namespace=namespace,
            max_entries=SEMANTIC_CONFIG.get("max_entries", 2000),
        )

        logger.info(f"3-Layer MemoryManager initialized (namespace={namespace})")

    def _maybe_migrate_legacy_structured(self, long_term_path: str) -> None:
        """
        One-time migration from legacy structured_memory.json if new file doesn't exist yet.
        """
        if os.path.exists(long_term_path):
            return
        legacy_path = os.path.join(BASE_DIR, "structured_memory.json")
        if not os.path.exists(legacy_path):
            return
        try:
            import json
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy = json.load(f)
            if isinstance(legacy, dict):
                self.long_term.data.update({k: v for k, v in legacy.items() if k in ("user", "preferences", "goals", "system_state")})
                self.long_term.save()
                logger.info(f"Migrated legacy structured memory from {legacy_path}")
        except Exception as exc:
            logger.warning(f"Legacy structured memory migration skipped: {exc}")

    # ================================================
    # Layer 1: Short-term memory
    # ================================================

    def _estimate_tokens(self, text: str) -> int:
        try:
            return max(1, int(self.model_manager.estimate_tokens(text)))
        except Exception:
            return max(1, int(len(text) / 4))

    def _short_term_tokens(self) -> int:
        return sum(self._estimate_tokens(m.get("content", "")) for m in self.short_term)

    def _truncate_short_term_by_tokens(self) -> None:
        while self.short_term and self._short_term_tokens() > self.short_term_max_tokens:
            self.short_term.popleft()

    def add_short_term_message(self, role: str, content: str) -> None:
        with self._lock:
            self.short_term.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self._truncate_short_term_by_tokens()

    def get_short_term_context(self) -> str:
        with self._lock:
            if not self.short_term:
                return ""
            lines = [f"{m['role'].capitalize()}: {m['content']}" for m in self.short_term]
            return "\n".join(lines)

    def get_short_term_messages(self) -> List[Dict]:
        with self._lock:
            return list(self.short_term)

    def clear_short_term(self) -> None:
        with self._lock:
            self.short_term.clear()

    def maybe_create_summary(self) -> bool:
        """
        Rolling summaries are intentionally disabled in 3-layer architecture.
        """
        return False

    def get_rolling_summary_context(self) -> str:
        return ""

    # ================================================
    # Layer 2: Long-term structured memory
    # ================================================

    def update_structured_memory(self, key: str, value: Any) -> None:
        self.long_term.set(key, value)

    def get_structured_memory(self, key: str = None) -> Any:
        if key is None:
            return self.long_term.get_all()
        return self.long_term.get(key)

    def _entry_importance(self, text: str) -> float:
        text_l = text.lower()
        score = 0.3
        if any(k in text_l for k in ("my name is", "born in", "my goal", "remember", "important", "deadline")):
            score += 0.4
        if any(k in text_l for k in ("prefer", "struggle", "love", "hate")):
            score += 0.2
        return min(1.0, score)

    def extract_profile_facts(self, message: str) -> bool:
        msg = message.lower()
        extracted_any = False

        name_match = re.search(
            r"\bmy name is\s+([A-Za-z][A-Za-z\s'\-]{0,60}?)(?:[.,!?]|$|\s+and\s+i\b)",
            message,
            flags=re.IGNORECASE,
        )
        if name_match:
            name = self._normalize_name(name_match.group(1))
            if name:
                self.update_structured_memory("user.name", name)
                extracted_any = True

        age_match = re.search(r"i(?:'m| am)?\s+(\d+)\s+years?\s+old", msg)
        if age_match:
            self.update_structured_memory("user.age", int(age_match.group(1)))
            extracted_any = True

        year_match = re.search(r"born in\s+(\d{4})", msg)
        if year_match:
            self.update_structured_memory("user.birth_year", int(year_match.group(1)))
            extracted_any = True

        color_match = re.search(r"my favorite colou?r is\s+(.+?)(?:\.|,|$)", msg)
        if color_match:
            self.update_structured_memory("preferences.favorite_color", color_match.group(1).strip())
            extracted_any = True

        prefer_match = re.search(r"i prefer\s+(.+?)(?:\.|,|$)", msg)
        if prefer_match:
            self.update_structured_memory("preferences.general_preference", prefer_match.group(1).strip())
            extracted_any = True

        like_match = re.search(r"i like\s+(.+?)(?:\.|,|$)", msg)
        if like_match:
            thing = like_match.group(1).strip()
            interests = self.get_structured_memory("preferences.interests") or []
            if isinstance(interests, list) and thing not in interests:
                interests.append(thing)
                self.update_structured_memory("preferences.interests", interests)
                extracted_any = True

        # Difficulty extraction
        struggle_match = re.search(
            r"\b(?:i(?:'m| am)?\s+(?:struggle|struggling|have trouble|have difficulties|have difficulty)\s+(?:in|with)\s+(.+?))(?:[.!?]|$)",
            msg
        )
        if struggle_match:
            subject_block = struggle_match.group(1).strip()
            difficulties = self.get_structured_memory("system_state.difficulties") or []
            for subject in self._split_subjects(subject_block):
                if isinstance(difficulties, list) and subject not in difficulties:
                    difficulties.append(subject)
            self.update_structured_memory("system_state.difficulties", difficulties)
            extracted_any = True

        find_diff_match = re.search(r"i find\s+(.+?)\s+difficult(?:[.!?]|$)", msg)
        if find_diff_match:
            subject_block = find_diff_match.group(1).strip()
            difficulties = self.get_structured_memory("system_state.difficulties") or []
            for subject in self._split_subjects(subject_block):
                if isinstance(difficulties, list) and subject not in difficulties:
                    difficulties.append(subject)
            self.update_structured_memory("system_state.difficulties", difficulties)
            extracted_any = True

        difficult_for_me_match = re.search(r"\b(.+?)\s+(?:is|are)\s+difficult(?:\s+for\s+me)?(?:[.!?]|$)", msg)
        if difficult_for_me_match:
            subject_block = difficult_for_me_match.group(1).strip()
            difficulties = self.get_structured_memory("system_state.difficulties") or []
            for subject in self._split_subjects(subject_block):
                if isinstance(difficulties, list) and subject not in difficulties:
                    difficulties.append(subject)
            self.update_structured_memory("system_state.difficulties", difficulties)
            extracted_any = True

        weak_in_match = re.search(r"\bi(?:'m| am)?\s+weak\s+(?:in|with)\s+(.+?)(?:[.!?]|$)", msg)
        if weak_in_match:
            subject_block = weak_in_match.group(1).strip()
            difficulties = self.get_structured_memory("system_state.difficulties") or []
            for subject in self._split_subjects(subject_block):
                if isinstance(difficulties, list) and subject not in difficulties:
                    difficulties.append(subject)
            self.update_structured_memory("system_state.difficulties", difficulties)
            extracted_any = True

        weak_subjects_match = re.search(r"\bmy\s+weak\s+subjects?\s+(?:are|is)\s+(.+?)(?:[.!?]|$)", msg)
        if weak_subjects_match:
            subject_block = weak_subjects_match.group(1).strip()
            difficulties = self.get_structured_memory("system_state.difficulties") or []
            for subject in self._split_subjects(subject_block):
                if isinstance(difficulties, list) and subject not in difficulties:
                    difficulties.append(subject)
            self.update_structured_memory("system_state.difficulties", difficulties)
            extracted_any = True

        if any(k in msg for k in ("my goal", "i want to", "i am preparing")):
            self.long_term.add_entry(message, metadata={"source": "profile_extraction"}, importance=self._entry_importance(message))
            extracted_any = True

        if extracted_any or self._looks_important_statement(message):
            self._add_long_term_fact(message)
            extracted_any = True

        return extracted_any

    def _format_structured_context(self) -> str:
        structured = self.get_structured_memory()
        if not isinstance(structured, dict):
            return ""

        lines = ["=== User Profile ==="]
        user = structured.get("user", {})
        prefs = structured.get("preferences", {})
        goals = structured.get("goals", [])
        system_state = structured.get("system_state", {})

        if user.get("name"):
            lines.append(f"Name: {user['name']}")
        if user.get("age"):
            lines.append(f"Age: {user['age']}")
        if user.get("birth_year"):
            lines.append(f"Birth Year: {user['birth_year']}")

        if prefs:
            lines.append("Preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")

        if goals:
            lines.append("Goals:")
            for g in goals[:5]:
                lines.append(f"- {g}")

        if system_state:
            lines.append("System State:")
            difficulties = system_state.get("difficulties")
            if isinstance(difficulties, list) and difficulties:
                lines.append("- difficulties: " + ", ".join(str(x) for x in difficulties))
            for k, v in system_state.items():
                if k == "difficulties":
                    continue
                lines.append(f"- {k}: {v}")

        # Add only top important long-term entries (avoid dumping full history)
        top_entries = self.long_term.top_entries(limit=4)
        if top_entries:
            lines.append("Important Long-Term Notes:")
            for entry in top_entries:
                lines.append(f"- {entry.get('text', '')}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _split_subjects(self, subject_block: str) -> List[str]:
        """
        Split difficulty/preference subject blocks into normalized items.
        Example: 'organic chemistry and thermodynamics' -> ['organic chemistry', 'thermodynamics']
        """
        if not subject_block:
            return []
        cleaned = subject_block.strip().strip(".")
        parts = re.split(r"\s*,\s*|\s+and\s+", cleaned)
        normalized = []
        for p in parts:
            item = p.strip().strip(".")
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    def _normalize_name(self, raw: str) -> str:
        if not raw:
            return ""
        candidate = raw.strip().strip(".,!?")
        candidate = re.split(
            r"\b(?:i am|i'm|i like|i prefer|and i|because|but)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip().strip(".,!?")
        return candidate

    def _looks_important_statement(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered or lowered.endswith("?"):
            return False
        cues = (
            "remember",
            "important",
            "my ",
            "i am ",
            "i'm ",
            "i have ",
            "i struggle",
            "i find",
            "i prefer",
            "i like",
            "i want",
            "my goal",
        )
        return any(c in lowered for c in cues)

    def _add_long_term_fact(self, text: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        try:
            entries = self.long_term.get("entries") or []
            if isinstance(entries, list):
                recent = entries[-20:]
                for entry in recent:
                    if (entry or {}).get("text", "").strip().lower() == normalized.lower():
                        return
            self.long_term.add_entry(
                normalized,
                metadata={"source": "fact_capture"},
                importance=self._entry_importance(normalized),
            )
        except Exception as exc:
            logger.warning(f"Long-term fact capture failed: {exc}")

    # ================================================
    # Layer 3: Vector memory
    # ================================================

    def add_semantic_memory(self, text: str, metadata: Dict = None) -> None:
        importance = self._entry_importance(text)
        ok = self.semantic.add_memory(text, metadata=metadata or {}, importance=importance)
        if not ok:
            logger.warning("Vector memory add failed; storing in long-term fallback")
            self.long_term.add_entry(text, metadata={"source": "vector_fallback"}, importance=importance)

    def search_semantic_memory(self, query: str, top_k: int = None) -> List[str]:
        if not SEMANTIC_CONFIG.get("enabled", True):
            return []
        k = int(top_k or DEFAULT_TOP_K)
        try:
            results = self.semantic.search(query, top_k=k)
            return [text for text, _score in results[:k]]
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

    def get_semantic_context(self, query: str) -> str:
        if not INCLUDE_SEMANTIC_IN_SEARCH:
            return ""
        results = self.search_semantic_memory(query)
        if not results:
            return ""
        return "=== Relevant Memories ===\n" + "\n".join(f"- {x}" for x in results)

    # ================================================
    # Context assembly + persistence
    # ================================================

    def build_full_context(self, current_query: str) -> Dict[str, str]:
        return {
            "short_term": self.get_short_term_context(),
            "rolling_summary": "",
            "semantic": self.get_semantic_context(current_query),
            "structured": self._format_structured_context(),
        }

    def save_all(self) -> None:
        self.long_term.save()
        self.semantic.save()
        self._sync_legacy_files()

    def add_user_message(self, content: str) -> None:
        self.add_short_term_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_short_term_message("assistant", content)

    def persist_structured_memory(self) -> None:
        self.long_term.save()
        self._sync_legacy_files()

    def persist_semantic_memory(self) -> None:
        self.semantic.save()
        self._sync_legacy_files()

    def _sync_legacy_files(self) -> None:
        """
        Backward compatibility:
        keep legacy files updated for existing workflows/inspection.
        """
        structured_legacy = os.path.join(BASE_DIR, "structured_memory.json")
        semantic_legacy = os.path.join(BASE_DIR, "semantic_memory.json")

        try:
            with open(structured_legacy, "w", encoding="utf-8") as f:
                json.dump(self.long_term.get_all(), f, indent=2)
        except Exception as exc:
            logger.warning(f"Legacy structured memory sync failed: {exc}")

        try:
            vector_rows = self.semantic.export_memories()
            with open(semantic_legacy, "w", encoding="utf-8") as f:
                json.dump(vector_rows, f, indent=2)
        except Exception as exc:
            logger.warning(f"Legacy semantic memory sync failed: {exc}")


class StructuredMemory(LongTermMemory):
    """
    Backward-compatible alias class.
    """
