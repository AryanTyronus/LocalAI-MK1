from core.config_loader import ConfigLoader
from services.ai_service import AIService
from services.memory_service import MemoryService
from core.tool_registry import ToolRegistry


def _register_default_tools():
    # Example safe tool: echo
    def _echo(params):
        return {'echo': params}

    ToolRegistry.register_tool('echo', 'Echo back parameters', {'type': 'object'}, _echo)

    # Example system-level tool (requires confirmation by default)
    def _open_app(params):
        app = params.get('app_name')
        return f"(simulated) opened {app}"

    ToolRegistry.register_tool('open_app', 'Open an application on the system (simulated)', {'app_name': 'string'}, _open_app)


class DependencyContainer:
    def __init__(self):
        self.config = ConfigLoader()

        # Create memory abstraction
        self.memory_service = MemoryService()

        # Inject into AIService
        # register some default tools
        _register_default_tools()

        self.ai_service = AIService(
            config=self.config,
            memory_service=self.memory_service
        )

    def get_ai_service(self):
        return self.ai_service