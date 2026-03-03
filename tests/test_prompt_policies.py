import unittest
from datetime import datetime

from core.generation_pipeline import GenerationPipeline


class _DummyMode:
    system_prompt = "Answer as a practical assistant."


class PromptPoliciesTest(unittest.TestCase):
    def test_system_message_contains_thinking_and_mode(self):
        obj = object.__new__(GenerationPipeline)
        obj._config = type("Cfg", (), {"personality_persistent_prompt": "Base personality."})()
        text = GenerationPipeline._build_system_message(
            obj,
            _DummyMode(),
            name="Aryan",
            birth_year=2005,
            age=21,
            current_year=2026,
            future_age=None,
            project="LocalAI",
            now_local=datetime.now().astimezone(),
            response_mode="technical",
        )
        self.assertIn("Internal Reasoning Policy (hidden):", text)
        self.assertIn("Never reveal internal reasoning", text)
        self.assertIn("Response Mode:", text)
        self.assertIn("Selected mode: technical", text)


if __name__ == "__main__":
    unittest.main()
