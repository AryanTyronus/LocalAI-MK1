"""
Routing layer for SYNAPSE's 3-tier intelligence hierarchy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from core.intent_classifier import IntentClassification
from models.config import MODEL_REGISTRY


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    complexity: str
    tier: str
    agent_name: str
    model_name: str
    mode: str
    use_planner: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ModelRouter:
    """Maps classified requests onto agents, models, and execution modes."""

    MODE_BY_AGENT = {
        "code_agent": "coding",
        "research_agent": "research",
        "memory_agent": "chat",
        "automation_agent": "agent",
        "file_agent": "agent",
    }

    MODEL_BY_AGENT = {
        "code_agent": MODEL_REGISTRY["coding"],
        "research_agent": MODEL_REGISTRY["research"],
        "memory_agent": MODEL_REGISTRY["embedding"],
        "automation_agent": MODEL_REGISTRY["research"],
        "file_agent": MODEL_REGISTRY["coding"],
    }

    def route(
        self,
        user_message: str,
        classification: IntentClassification,
        mode_hint: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> RouteDecision:
        del user_message
        del options

        agent_name = classification.agent
        model_name = self.MODEL_BY_AGENT.get(agent_name, MODEL_REGISTRY["research"])
        mode = self.MODE_BY_AGENT.get(agent_name, "chat")
        if mode_hint and mode_hint != "chat":
            mode = mode_hint
        use_planner = False
        tier = "tier2"

        if classification.complexity == "high":
            model_name = MODEL_REGISTRY["reasoning"]
            use_planner = True
            tier = "tier3"
        elif classification.agent in ("memory_agent",) and classification.complexity == "low":
            tier = "tier1"
        elif classification.tier:
            tier = classification.tier

        return RouteDecision(
            intent=classification.intent,
            complexity=classification.complexity,
            tier=tier,
            agent_name=agent_name,
            model_name=model_name,
            mode=mode,
            use_planner=use_planner,
        )
