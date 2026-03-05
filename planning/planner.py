"""
Structured planning layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Plan:
    steps: List[str]
    reason: str
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "steps": list(self.steps),
            "reason": self.reason,
            "confidence": float(self.confidence),
        }


class Planner:
    """
    Lightweight deterministic planner.
    """

    def plan(self, user_query: str) -> Plan:
        text = (user_query or "").strip().lower()
        if not text:
            return Plan(steps=["respond"], reason="empty_query", confidence=0.2)

        steps: List[str] = []
        if any(k in text for k in ("latest", "news", "current", "today")):
            steps.extend(["search_web", "read_page"])
        if any(k in text for k in ("file", "read", "open")):
            steps.append("read_file")
        if any(k in text for k in ("write", "save", "create file")):
            steps.append("write_file")
        if any(k in text for k in ("test", "run code", "execute python")):
            steps.append("run_tests")

        if not steps:
            steps = ["respond"]
            reason = "direct_response"
            confidence = 0.7
        else:
            steps.append("summarize")
            reason = "task_or_tool_intent"
            confidence = 0.84

        # Stable dedupe while preserving order.
        deduped = list(dict.fromkeys(steps))
        return Plan(steps=deduped, reason=reason, confidence=confidence)

