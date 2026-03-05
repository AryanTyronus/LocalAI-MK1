"""
Safe, reversible prompt optimization templates.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from core.config import BASE_DIR
from core.logger import logger

DEFAULT_TEMPLATES = {
    "empty_response": "Return a direct answer with complete content in plain text.",
    "too_short_for_question": "Answer with sufficient detail for the user question.",
    "analytical_too_brief": "Provide stepwise reasoning and a final concise conclusion.",
    "analytical_lacks_reasoning_markers": "Include key intermediate logic before the final result.",
    "echoed_user_prompt": "Do not repeat the user question; provide a true answer.",
    "reasoning_leakage_marker": "Do not expose internal reasoning; provide only final answer.",
}


class PromptOptimizer:
    def __init__(self, store_path: Optional[str] = None):
        self._store_path = store_path or os.path.join(
            BASE_DIR, "prompt_optimization", "optimized_prompts.json"
        )
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if not os.path.exists(self._store_path):
            return {"enabled": True, "templates": dict(DEFAULT_TEMPLATES)}
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("invalid_state")
            templates = data.get("templates", {})
            merged = dict(DEFAULT_TEMPLATES)
            if isinstance(templates, dict):
                for key, value in templates.items():
                    if isinstance(key, str) and isinstance(value, str) and value.strip():
                        merged[key] = value.strip()
            return {
                "enabled": bool(data.get("enabled", True)),
                "templates": merged,
            }
        except Exception as exc:
            logger.warning(f"PromptOptimizer load failed, using defaults: {exc}")
            return {"enabled": True, "templates": dict(DEFAULT_TEMPLATES)}

    def _save_state(self) -> None:
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=True)
        except Exception as exc:
            logger.warning(f"PromptOptimizer save failed: {exc}")

    def is_enabled(self) -> bool:
        return bool(self._state.get("enabled", True))

    def set_enabled(self, enabled: bool) -> None:
        self._state["enabled"] = bool(enabled)
        self._save_state()

    def get_guidance(self, reason: str) -> str:
        templates = self._state.get("templates", {})
        if not isinstance(templates, dict):
            return ""
        return str(templates.get(reason, "")).strip()

    def update_template(self, reason: str, guidance: str) -> None:
        if not (reason or "").strip() or not (guidance or "").strip():
            return
        templates = self._state.setdefault("templates", {})
        if not isinstance(templates, dict):
            self._state["templates"] = {}
            templates = self._state["templates"]
        templates[str(reason).strip()] = str(guidance).strip()
        self._save_state()

    def learn_from_outcome(self, reason: str, resolved: bool) -> None:
        """
        Conservative optimization:
        only reinforce existing template metadata when reflection succeeds.
        No autonomous risky rewriting.
        """
        if resolved and self.get_guidance(reason):
            # Touch state to persist deterministic template baseline.
            self._save_state()

