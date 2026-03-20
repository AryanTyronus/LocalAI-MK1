from __future__ import annotations

import re

from agents.base_agent import BaseAgent


class FileAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="file_agent", **kwargs)

    def execute(self, task):
        message = (task.get("user_message") or "").strip()
        if self.tool_registry and self._looks_like_file_read(message):
            filepath = self._extract_filepath(message)
            if filepath:
                future = self.submit_tool("file_reader", {"filepath": filepath}, require_confirmation=False)
                result = future.result(timeout=5)
                return {
                    "text": str(result),
                    "agent": self.name,
                    "model_name": task.get("model_name"),
                    "mode": "agent",
                }

        task = dict(task)
        task.setdefault("mode", "agent")
        return self._pipeline_generate(task)

    def stream(self, task):
        task = dict(task)
        task.setdefault("mode", "agent")
        yield from self._pipeline_stream(task)

    def _looks_like_file_read(self, message: str) -> bool:
        lowered = message.lower()
        return "read" in lowered and ("file" in lowered or "/" in message or "." in message)

    def _extract_filepath(self, message: str):
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", message)
        if quoted:
            return quoted[0]
        match = re.search(r"(?:file\s+)?([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)", message)
        if match:
            return match.group(1)
        return None
