"""
Prompt/reflection outcome tracker.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

from core.config import BASE_DIR
from core.logger import logger


class PromptTracker:
    def __init__(self, log_path: Optional[str] = None):
        self._log_path = log_path or os.path.join(
            BASE_DIR, "prompt_optimization", "prompt_events.jsonl"
        )
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def track_event(self, event: Dict) -> None:
        payload = dict(event or {})
        payload.setdefault("timestamp", datetime.now().astimezone().isoformat())
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning(f"PromptTracker write failed: {exc}")

