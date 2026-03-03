import unittest

from core.request_policies import (
    classify_intent,
    classify_response_mode,
    build_context_policy,
)


class RequestPoliciesTest(unittest.TestCase):
    def test_intent_confidence_outputs(self):
        cases = [
            ("latest news today", "live_data"),
            ("show system status and cpu usage", "system_query"),
            ("run python to print hello", "tool_request"),
            ("do you remember my name", "memory_recall"),
            ("solve this integral step by step", "analytical_problem"),
            ("hello there", "general_chat"),
        ]
        for prompt, expected in cases:
            intent, confidence = classify_intent(prompt)
            self.assertEqual(intent, expected)
            self.assertGreaterEqual(confidence, 0.01)
            self.assertLessEqual(confidence, 0.99)

    def test_response_mode_style_override(self):
        mode, confidence, cleaned = classify_response_mode(
            "/style technical explain this bug quickly", "general_chat"
        )
        self.assertEqual(mode, "technical")
        self.assertGreaterEqual(confidence, 0.95)
        self.assertEqual(cleaned, "explain this bug quickly")

    def test_response_mode_basics(self):
        mode, _, _ = classify_response_mode("Explain this in detail", "general_chat")
        self.assertEqual(mode, "detailed")

        mode, _, _ = classify_response_mode("hey thanks", "general_chat")
        self.assertEqual(mode, "casual")

        mode, _, _ = classify_response_mode("debug this api endpoint", "tool_request")
        self.assertEqual(mode, "technical")

    def test_context_policy_matrix(self):
        analytical = build_context_policy("solve this integral", "analytical_problem")
        self.assertFalse(analytical["use_web"])
        self.assertFalse(analytical["include_documents"])

        live = build_context_policy("latest news today", "live_data")
        self.assertTrue(live["use_web"])
        self.assertFalse(live["include_documents"])

        memory = build_context_policy("do you remember my name", "memory_recall")
        self.assertFalse(memory["use_web"])
        self.assertFalse(memory["include_documents"])
        self.assertTrue(memory["memory_enabled"])

        mixed = build_context_policy("do you remember my name and latest news today", "memory_recall")
        self.assertTrue(mixed["use_web"])
        self.assertIn("memory", mixed["sources"])
        self.assertIn("web", mixed["sources"])


if __name__ == "__main__":
    unittest.main()
