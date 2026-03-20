from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Dict, Generator, Optional

from core.logger import logger


class BaseAgent:
    """Unified agent interface for tier-2 specialist workers."""

    _tool_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="synapse_tool")

    def __init__(
        self,
        name: str,
        model_manager,
        pipeline=None,
        memory_manager=None,
        document_manager=None,
        tool_registry=None,
    ):
        self.name = name
        self.model_manager = model_manager
        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.document_manager = document_manager
        self.tool_registry = tool_registry

    def execute(self, task: Dict) -> Dict:
        raise NotImplementedError

    def stream(self, task: Dict) -> Generator[Dict, None, None]:
        result = self.execute(task)
        yield {"content": result.get("text", "")}
        yield {"done": True, "agent": self.name, "model_name": result.get("model_name")}

    def submit_tool(self, tool_name: str, params: Dict, require_confirmation: bool = False) -> Future:
        return self._tool_pool.submit(
            self.tool_registry.execute_tool,
            tool_name,
            params,
            require_confirmation,
        )

    def _generation_context(self, model_name: Optional[str]):
        if model_name and hasattr(self.model_manager, "activate_model"):
            return self.model_manager.activate_model(model_name)
        return nullcontext()

    def _pipeline_generate(self, task: Dict) -> Dict:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} has no generation pipeline configured")

        options = dict(task.get("options") or {})
        mode = task.get("mode", "chat")
        user_message = task.get("user_message", "")
        model_name = task.get("model_name")

        with self._generation_context(model_name):
            text = self.pipeline.generate(user_message, mode, options=options)
        logger.info("Agent '%s' completed generation with model '%s'", self.name, model_name)
        return {"text": text, "agent": self.name, "model_name": model_name, "mode": mode}

    def _pipeline_stream(self, task: Dict) -> Generator[Dict, None, None]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} has no generation pipeline configured")

        options = dict(task.get("options") or {})
        mode = task.get("mode", "chat")
        user_message = task.get("user_message", "")
        model_name = task.get("model_name")

        with self._generation_context(model_name):
            for item in self.pipeline.run_stream(user_message, mode, options=options):
                yield item
