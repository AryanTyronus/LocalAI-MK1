import unittest
from unittest.mock import patch

from core.model_adapter import OllamaModelAdapter
from core.model_manager import FakeModelManager


class OllamaAdapterTest(unittest.TestCase):
    def test_ollama_adapter_generates_and_reports_backend(self):
        manager = FakeModelManager(reason="test")
        adapter = OllamaModelAdapter(manager)

        with patch.object(
            OllamaModelAdapter,
            "_post_json",
            return_value={"response": "hello from ollama"},
        ):
            result = adapter.generate("hi", max_tokens=16)

        self.assertEqual(result.text, "hello from ollama")
        self.assertEqual(result.model_name, "qwen3:8b")
        self.assertEqual(adapter.get_backend_info()["backend"], "ollama")


if __name__ == "__main__":
    unittest.main()
