"""
Request routing policies for intent, response mode, and context selection.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

LIVE_DATA_KEYWORDS = [
    "latest",
    "today",
    "current",
    "2026",
    "right now",
    "news",
    "stock",
    "price",
    "score",
    "who is",
]

SYSTEM_QUERY_PATTERNS = [
    r"\b(system|backend)\s+(status|health|metrics|diagnostics|logs?)\b",
    r"\b(cpu|ram|temperature|thermal|uptime)\b",
    r"\b(session|context)\s+(metrics|size)\b",
    r"\b(show|check|get)\s+(system|activity)\s+log(s)?\b",
]

MEMORY_RECALL_PATTERNS = [
    r"\bwhat do you remember\b",
    r"\bdo you remember\b",
    r"\brecall\b",
    r"\bmy name\b",
    r"\bwhat did i (tell|say)\b",
]

ANALYTICAL_PATTERNS = [
    r"\b(solve|derive|prove|analyze|calculate|optimize)\b",
    r"\b(step by step|equation|integral|derivative|probability|complexity)\b",
]

TOOL_REQUEST_PATTERNS = [
    r"^/tool\b",
    r"\b(run|execute)\s+python\b",
    r"\b(read|show)\s+file\b",
    r"\b(weather|stock|quote|ticker)\b",
]

STYLE_OVERRIDE_RE = re.compile(
    r"^\s*(?:/style\s+|style\s*:\s*)(concise|detailed|analytical|casual|technical)\b[:\-\s]*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def needs_live_data(prompt: str) -> bool:
    normalized = (prompt or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in LIVE_DATA_KEYWORDS)


def classify_intent(prompt: str) -> Tuple[str, float]:
    """
    Return (intent, confidence).
    """
    text = (prompt or "").strip().lower()
    if not text:
        return "general_chat", 0.35

    scores = {
        "general_chat": 0.35,
        "live_data": 0.0,
        "system_query": 0.0,
        "tool_request": 0.0,
        "memory_recall": 0.0,
        "analytical_problem": 0.0,
    }

    def add_score(intent: str, patterns, weight: float) -> None:
        for pattern in patterns:
            if re.search(pattern, text):
                scores[intent] += weight

    add_score("system_query", SYSTEM_QUERY_PATTERNS, 0.34)
    add_score("memory_recall", MEMORY_RECALL_PATTERNS, 0.34)
    add_score("tool_request", TOOL_REQUEST_PATTERNS, 0.36)
    add_score("analytical_problem", ANALYTICAL_PATTERNS, 0.34)

    if needs_live_data(text):
        scores["live_data"] += 0.36
    if re.search(r"\b(news|headline|breaking|update)\b", text):
        scores["live_data"] += 0.18

    top_intent = max(scores, key=scores.get)
    confidence = max(0.01, min(0.99, scores[top_intent]))
    return top_intent, confidence


def classify_response_mode(prompt: str, intent: str) -> Tuple[str, float, str]:
    """
    Return (mode, confidence, cleaned_prompt).
    """
    raw = (prompt or "").strip()
    if not raw:
        return "concise", 0.35, raw

    m = STYLE_OVERRIDE_RE.match(raw)
    if m:
        mode = m.group(1).strip().lower()
        cleaned = (m.group(2) or "").strip()
        return mode, 0.99, (cleaned or raw)

    text = raw.lower()
    scores = {
        "concise": 0.35,
        "detailed": 0.0,
        "analytical": 0.0,
        "casual": 0.0,
        "technical": 0.0,
    }

    if intent == "analytical_problem":
        scores["analytical"] += 0.62
    if intent == "live_data":
        scores["concise"] += 0.24

    if any(k in text for k in ("code", "debug", "api", "endpoint", "stack trace", "algorithm", "architecture", "refactor")):
        scores["technical"] += 0.58
    if re.search(r"\b(in detail|detailed|deep dive|comprehensive|step by step)\b", text):
        scores["detailed"] += 0.6
    if re.search(r"\b(hey|hi|thanks|lol|bro|buddy)\b", text):
        scores["casual"] += 0.56
    if any(k in text for k in ("latest", "today", "news", "status", "quick", "brief", "summary")):
        scores["concise"] += 0.32
    if len(text.split()) >= 26:
        scores["detailed"] += 0.26

    mode = max(scores, key=scores.get)
    confidence = max(0.01, min(0.99, scores[mode]))
    return mode, confidence, raw


def build_context_policy(prompt: str, intent: str) -> Dict:
    text = (prompt or "").strip().lower()
    has_live_signal = needs_live_data(text) or bool(re.search(r"\b(news|headline|update|breaking)\b", text))
    has_memory_signal = bool(any(re.search(pattern, text) for pattern in MEMORY_RECALL_PATTERNS))
    is_math_like = intent == "analytical_problem" and bool(
        re.search(r"\b(math|equation|integral|derivative|algebra|calculus|probability|solve|derive|calculate)\b", text)
    )
    is_news_like = bool(re.search(r"\b(news|headline|breaking|update|latest)\b", text))
    mixed_memory_live = has_live_signal and has_memory_signal

    policy = {
        "use_web": False,
        "memory_enabled": True,
        "include_documents": True,
        "reason": "default",
        "sources": ["memory", "documents"],
        "confidence": 0.65,
    }

    if mixed_memory_live:
        return {
            "use_web": True,
            "memory_enabled": True,
            "include_documents": False,
            "reason": "mixed_memory_web",
            "sources": ["memory", "web"],
            "confidence": 0.9,
        }
    if intent == "memory_recall":
        return {
            "use_web": False,
            "memory_enabled": True,
            "include_documents": False,
            "reason": "memory_only",
            "sources": ["memory"],
            "confidence": 0.9,
        }
    if is_math_like:
        return {
            "use_web": False,
            "memory_enabled": True,
            "include_documents": False,
            "reason": "analytical_no_web",
            "sources": ["memory"],
            "confidence": 0.88,
        }
    if intent == "live_data" or is_news_like:
        return {
            "use_web": True,
            "memory_enabled": True,
            "include_documents": False,
            "reason": "news_live_web",
            "sources": ["web"],
            "confidence": 0.88,
        }
    if intent == "system_query":
        return {
            "use_web": False,
            "memory_enabled": False,
            "include_documents": False,
            "reason": "system_internal",
            "sources": ["system"],
            "confidence": 0.95,
        }
    if intent == "tool_request":
        return {
            "use_web": False,
            "memory_enabled": False,
            "include_documents": False,
            "reason": "tool_direct",
            "sources": ["tool"],
            "confidence": 0.9,
        }
    return policy


def sanitize_model_reply(reply: str) -> str:
    """
    Remove common accidental chain-of-thought leakage markers.
    """
    text = (reply or "").strip()
    if not text:
        return text

    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?im)^(reasoning|chain[- ]of[- ]thought|thought process)\s*:\s*.*$", "", text)

    m = re.search(r"(?is)\bfinal answer\s*:\s*(.+)$", text)
    if m:
        text = m.group(1).strip()

    # Remove excessive blank lines introduced by scrubbing.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
