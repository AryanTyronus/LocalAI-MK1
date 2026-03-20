"""
Central model registry for SYNAPSE.

This file is intentionally additive: aliases can point to the same local model
until dedicated local weights are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


def _env_or_default(name: str, fallback: str) -> str:
    value = os.getenv(name, "").strip()
    return value or fallback


_DEFAULT_LOCAL_MLX = _env_or_default(
    "SYNAPSE_DEFAULT_MLX_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    role: str
    backend: str
    model_ref: str
    supports_generation: bool = True
    supports_embeddings: bool = False
    lazy: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, str] = {
    "reasoning": "qwen3",
    "coding": "nemotron_openrouter",
    "research": "mistral",
    "embedding": "bge",
    "stt": "whisper",
    "tts": "piper",
    "router": "phi3_mini",
}

NEMOTRON_OPENROUTER = "nvidia/nemotron-3-super-120b-a12b:free"



MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen3": ModelSpec(
        alias="qwen3",
        role="reasoning",
        backend="mlx",
        model_ref=_env_or_default("SYNAPSE_REASONING_MODEL", _DEFAULT_LOCAL_MLX),
        metadata={"tier": "tier3"},
    ),
    "nemotron_openrouter": ModelSpec(
        alias="nemotron_openrouter",
        role="coding",
        backend="openrouter",
        model_ref=NEMOTRON_OPENROUTER,
        metadata={"tier": "openrouter", "provider": "nvidia"},
    ),
    "mistral": ModelSpec(
        alias="mistral",
        role="research",
        backend="mlx",
        model_ref=_env_or_default("SYNAPSE_RESEARCH_MODEL", _DEFAULT_LOCAL_MLX),
        metadata={"tier": "tier2"},
    ),
    "bge": ModelSpec(
        alias="bge",
        role="embedding",
        backend="sentence-transformers",
        model_ref=_env_or_default("SYNAPSE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        supports_generation=False,
        supports_embeddings=True,
        metadata={"tier": "memory"},
    ),
    "e5": ModelSpec(
        alias="e5",
        role="embedding",
        backend="sentence-transformers",
        model_ref=_env_or_default("SYNAPSE_EMBEDDING_MODEL_E5", "intfloat/e5-small-v2"),
        supports_generation=False,
        supports_embeddings=True,
        metadata={"tier": "memory"},
    ),
    "whisper": ModelSpec(
        alias="whisper",
        role="speech_to_text",
        backend="placeholder",
        model_ref=_env_or_default("SYNAPSE_STT_MODEL", "openai/whisper-small"),
        supports_generation=False,
        metadata={"tier": "voice"},
    ),
    "piper": ModelSpec(
        alias="piper",
        role="text_to_speech",
        backend="placeholder",
        model_ref=_env_or_default("SYNAPSE_TTS_MODEL", "piper/en_US-lessac-medium"),
        supports_generation=False,
        metadata={"tier": "voice"},
    ),
    "phi3_mini": ModelSpec(
        alias="phi3_mini",
        role="router",
        backend="rule",
        model_ref=_env_or_default("SYNAPSE_ROUTER_MODEL", "rule-based-router"),
        supports_generation=False,
        metadata={"tier": "tier1"},
    ),
}


ALIASES: Dict[str, str] = {
    "reasoning": MODEL_REGISTRY["reasoning"],
    "coding": MODEL_REGISTRY["coding"],
    "research": MODEL_REGISTRY["research"],
    "embedding": MODEL_REGISTRY["embedding"],
    "stt": MODEL_REGISTRY["stt"],
    "tts": MODEL_REGISTRY["tts"],
    "router": MODEL_REGISTRY["router"],
    "qwen": "qwen3",
    "qwen3": "qwen3",
    "nemotron": "nemotron_openrouter",
    "nemotron_openrouter": "nemotron_openrouter",
    "openrouter": "nemotron_openrouter",
    "mistral": "mistral",
    "bge": "bge",
    "e5": "e5",
    "whisper": "whisper",
    "piper": "piper",
    "phi3": "phi3_mini",
    "phi3_mini": "phi3_mini",
}


def resolve_model_alias(name: str) -> str:
    normalized = (name or MODEL_REGISTRY["reasoning"]).strip().lower()
    return ALIASES.get(normalized, normalized)
