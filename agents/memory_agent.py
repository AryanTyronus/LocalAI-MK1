from __future__ import annotations

from agents.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="memory_agent", **kwargs)

    def execute(self, task):
        query = (task.get("routing_message") or task.get("user_message") or "").strip()
        if not self.memory_manager:
            return {"text": "Memory subsystem is unavailable.", "agent": self.name, "model_name": task.get("model_name")}

        sections = []
        build_full_context = getattr(self.memory_manager, "build_full_context", None)
        if callable(build_full_context):
            context = build_full_context(query)
            rendered = self._render_context_block(context)
            if rendered:
                sections.append(rendered)

        search_semantic = getattr(self.memory_manager, "search_semantic_memory", None)
        if callable(search_semantic):
            memories = search_semantic(query)
            if memories:
                sections.append("Relevant memories:\n" + "\n".join(f"- {item}" for item in memories[:5]))

        if not sections:
            sections.append("No relevant memory found.")

        return {"text": "\n\n".join(sections), "agent": self.name, "model_name": task.get("model_name"), "mode": "chat"}

    def _render_context_block(self, context) -> str:
        if not context:
            return ""
        if isinstance(context, str):
            return context.strip()
        if isinstance(context, dict):
            ordered = []
            for key in ("structured", "short_term", "semantic", "rolling_summary"):
                value = context.get(key)
                if isinstance(value, str) and value.strip():
                    ordered.append(value.strip())
            for key, value in context.items():
                if key in {"structured", "short_term", "semantic", "rolling_summary"}:
                    continue
                if isinstance(value, str) and value.strip():
                    ordered.append(value.strip())
            return "\n\n".join(ordered).strip()
        if isinstance(context, (list, tuple)):
            return "\n".join(str(item) for item in context if item).strip()
        return str(context).strip()
