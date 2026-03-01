import json
from typing import Callable, Dict, List
from core.logger import logger


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        schema: Dict,
        func: Callable,
        category: str = "tools",
        human_name: str = "",
        example_usage: str = "",
    ):
        self.name = name
        self.description = description
        self.schema = schema or {}
        self.func = func
        self.category = (category or "tools").strip().lower()
        self.human_name = (human_name or name).strip()
        self.example_usage = (example_usage or "").strip()


class ToolRegistry:
    _tools: Dict[str, Tool] = {}

    @classmethod
    def register_tool(
        cls,
        name: str,
        description: str,
        schema: Dict,
        func: Callable,
        category: str = "tools",
        human_name: str = "",
        example_usage: str = "",
    ):
        cls._tools[name] = Tool(
            name,
            description,
            schema,
            func,
            category=category,
            human_name=human_name,
            example_usage=example_usage,
        )
        logger.info(f"Registered tool: {name}")

    @classmethod
    def get_tool(cls, name: str):
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> List[Dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": dict(tool.schema),
                "category": tool.category,
                "human_name": tool.human_name,
                "example_usage": tool.example_usage,
            }
            for tool in sorted(cls._tools.values(), key=lambda t: t.name)
        ]

    @classmethod
    def execute_tool(cls, name: str, parameters: Dict, require_confirmation: bool = True):
        tool = cls.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        # Safety confirmation layer
        if require_confirmation:
            # Caller must explicitly confirm via parameter '_confirmed': True
            if not parameters.get('_confirmed'):
                return {'status': 'requires_confirmation', 'message': f"Execution of {name} requires confirmation."}

        # Remove the internal flag before executing
        if '_confirmed' in parameters:
            del parameters['_confirmed']

        # Execute tool function
        try:
            result = tool.func(parameters)
            return {'status': 'ok', 'result': result}
        except Exception as e:
            logger.exception(f"Tool {name} execution failed")
            return {'status': 'error', 'error': str(e)}
