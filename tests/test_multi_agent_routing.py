import unittest

from core.intent_classifier import IntentClassifier
from core.model_manager import FakeModelManager
from core.orchestrator import SynapseOrchestrator
from core.router import ModelRouter
from models.config import MODEL_REGISTRY


class StubPipeline:
    def __init__(self):
        self.calls = []

    def generate(self, user_message, mode, options=None):
        self.calls.append(("generate", user_message, mode, options or {}))
        return f"{mode}:{user_message}"

    def run_stream(self, user_message, mode, options=None):
        self.calls.append(("stream", user_message, mode, options or {}))
        yield {"content": f"{mode}:{user_message}"}
        yield {"done": True}

    def get_last_turn_meta(self):
        return {"memory_updated": False}


class StubMemoryManager:
    def build_full_context(self, query):
        return f"context:{query}"

    def search_semantic_memory(self, query):
        return [f"memory:{query}"]


class MultiAgentRoutingTest(unittest.TestCase):
    def test_intent_classifier_returns_structured_routing(self):
        result = IntentClassifier.classify_request("Please debug this Python function and refactor it")
        self.assertEqual(result.intent, "coding")
        self.assertEqual(result.agent, "code_agent")
        self.assertIn(result.complexity, {"medium", "high"})

    def test_router_uses_reasoning_model_for_high_complexity(self):
        classification = IntentClassifier.classify_request(
            "Design a complex multi-step architecture migration plan for this system"
        )
        route = ModelRouter().route("complex task", classification, mode_hint="chat")
        self.assertEqual(route.model_name, MODEL_REGISTRY["reasoning"])
        self.assertTrue(route.use_planner)
        self.assertEqual(route.tier, "tier3")

    def test_synapse_orchestrator_dispatches_memory_agent(self):
        orchestrator = SynapseOrchestrator(
            model_manager=FakeModelManager(),
            pipeline=StubPipeline(),
            memory_manager=StubMemoryManager(),
            document_manager=None,
            tool_registry=None,
        )
        result = orchestrator.handle_request("remember that I like physics", mode="chat")
        self.assertEqual(result["route"]["agent_name"], "memory_agent")
        self.assertIn("context:remember", result["text"])


if __name__ == "__main__":
    unittest.main()
