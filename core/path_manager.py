"""Centralized path manager with bootstrap helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PathBundle:
    base_dir: str
    memory_dir: str
    logs_dir: str
    memory_file: str
    semantic_file: str
    facts_file: str


class PathManager:
    def __init__(self, base_dir: str):
        self._base_dir = os.path.abspath(base_dir)
        self._memory_dir = os.path.join(self._base_dir, "memory")
        self._logs_dir = os.path.join(self._base_dir, "logs")

    @property
    def bundle(self) -> PathBundle:
        return PathBundle(
            base_dir=self._base_dir,
            memory_dir=self._memory_dir,
            logs_dir=self._logs_dir,
            memory_file=os.path.join(self._memory_dir, "long_term.json"),
            semantic_file=os.path.join(self._memory_dir, "semantic_memory.json"),
            facts_file=os.path.join(self._memory_dir, "facts.json"),
        )

    def ensure_directories(self) -> None:
        os.makedirs(self._memory_dir, exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self._memory_dir, "knowledge_index"), exist_ok=True)

    def bootstrap_memory_files(self) -> Dict[str, bool]:
        self.ensure_directories()
        created = {
            "memory_file": self._ensure_json_file(self.bundle.memory_file, {
                "user": {},
                "preferences": {},
                "goals": {},
                "system_state": {},
                "entries": [],
            }),
            "semantic_file": self._ensure_json_file(self.bundle.semantic_file, []),
            "facts_file": self._ensure_json_file(self.bundle.facts_file, []),
        }
        return created

    def _ensure_json_file(self, path: str, default_obj):
        if os.path.exists(path):
            return False
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, indent=2)
        return True
