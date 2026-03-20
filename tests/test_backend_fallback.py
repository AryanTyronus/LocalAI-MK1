import os
import unittest


class BackendFallbackTest(unittest.TestCase):
    def setUp(self):
        self._old_force = os.environ.get("LOCALAI_FORCE_FALLBACK")
        self._old_retry = os.environ.get("LOCALAI_RETRY_MLX")
        self._old_dev_mode = os.environ.get("LOCALAI_DEV_MODE")
        os.environ["LOCALAI_FORCE_FALLBACK"] = "1"
        os.environ.pop("LOCALAI_RETRY_MLX", None)
        os.environ.pop("LOCALAI_DEV_MODE", None)

    def tearDown(self):
        if self._old_force is None:
            os.environ.pop("LOCALAI_FORCE_FALLBACK", None)
        else:
            os.environ["LOCALAI_FORCE_FALLBACK"] = self._old_force
        if self._old_retry is None:
            os.environ.pop("LOCALAI_RETRY_MLX", None)
        else:
            os.environ["LOCALAI_RETRY_MLX"] = self._old_retry
        if self._old_dev_mode is None:
            os.environ.pop("LOCALAI_DEV_MODE", None)
        else:
            os.environ["LOCALAI_DEV_MODE"] = self._old_dev_mode

    def test_model_manager_uses_safe_fallback_when_forced(self):
        from core.model_manager import FakeModelManager, ModelManager

        original_instance = ModelManager._instance
        try:
            ModelManager._instance = None
            manager = ModelManager.get_instance()
            self.assertIsInstance(manager, FakeModelManager)
            backend = manager.get_backend_info()
            self.assertEqual(backend["backend"], "safe-fallback")
            self.assertIn("LOCALAI_FORCE_FALLBACK", backend["reason"])
        finally:
            ModelManager._instance = original_instance

    def test_chat_and_status_work_in_safe_fallback_mode(self):
        import app as app_module
        from core.orchestrator import AppOrchestrator

        original_orchestrator = app_module._orchestrator
        original_ai_service = app_module._ai_service
        original_initialized = app_module._synapse_initialized
        original_ready = app_module._synapse_ready
        original_initializing = app_module._synapse_initializing

        try:
            app_module._orchestrator = AppOrchestrator()
            app_module._ai_service = None
            app_module._synapse_initialized = False
            app_module._synapse_ready = False
            app_module._synapse_initializing = False

            with app_module.app.test_client() as client:
                chat = client.post("/chat", json={"message": "Hello", "mode": "chat"})
                self.assertEqual(chat.status_code, 200)
                payload = chat.get_json() or {}
                self.assertTrue(payload.get("response"))

                status = client.get("/system/status")
                self.assertEqual(status.status_code, 200)
                status_payload = status.get_json() or {}
                self.assertEqual(status_payload.get("backend", {}).get("backend"), "safe-fallback")
                self.assertIn(status_payload.get("status"), {"SAFE MODE", "STANDBY"})
        finally:
            app_module._orchestrator = original_orchestrator
            app_module._ai_service = original_ai_service
            app_module._synapse_initialized = original_initialized
            app_module._synapse_ready = original_ready
            app_module._synapse_initializing = original_initializing


if __name__ == "__main__":
    unittest.main()
