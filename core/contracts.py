"""Typed data contracts for SYNAPSE runtime interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class MessageRecord(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str


class ToolCallPayload(TypedDict):
    tool_name: str
    parameters: Dict[str, Any]


class TelemetryPayload(TypedDict, total=False):
    timestamp: str
    state: str
    tokens_per_sec: float
    context_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    memory_hits: int
    correction_count: int
    latency_ms: float
    session_id: str


class MemoryFactRecord(TypedDict, total=False):
    subject: str
    predicate: str
    object: str
    timestamp: str
    source: str
    embedding: List[float]


class GenerationConfig(TypedDict, total=False):
    mode: str
    temperature: float
    max_tokens: int
    response_mode: str
    memory_enabled: bool
    include_documents: bool
    reflection_enabled: bool
    enable_self_correction: bool


class GenerationMeta(TypedDict, total=False):
    memory_updated: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_tokens: int
    correction_count: int
    memory_hits: int


class RuntimeEvent(TypedDict, total=False):
    type: str
    timestamp: str
    payload: Dict[str, Any]
