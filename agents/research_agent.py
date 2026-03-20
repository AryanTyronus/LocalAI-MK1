from __future__ import annotations

from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="research_agent", **kwargs)

    def execute(self, task):
        task = dict(task)
        task.setdefault("mode", "research")
        return self._pipeline_generate(task)

    def stream(self, task):
        task = dict(task)
        task.setdefault("mode", "research")
        yield from self._pipeline_stream(task)
