from __future__ import annotations

from agents.base_agent import BaseAgent


class CodeAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="code_agent", **kwargs)

    def execute(self, task):
        task = dict(task)
        task.setdefault("mode", "coding")
        return self._pipeline_generate(task)

    def stream(self, task):
        task = dict(task)
        task.setdefault("mode", "coding")
        yield from self._pipeline_stream(task)
