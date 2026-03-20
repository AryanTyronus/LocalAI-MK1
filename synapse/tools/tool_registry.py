"""Tool Registry - Manual function-calling system for DeepSeek."""

import subprocess
import logging
import json
import webbrowser
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a callable tool."""
    name: str
    description: str
    handler: Callable
    parameters: Dict[str, str]  # param_name -> description


class ToolRegistry:
    """
    Registry of available tools for SYNAPSE.
    Manually parses LLM JSON responses and executes tools.
    """

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        handler: Callable,
        description: str,
        parameters: Optional[Dict[str, str]] = None,
    ):
        """
        Register a new tool.

        Args:
            name: Tool name (must be lowercase, no spaces)
            handler: Callable that executes the tool
            description: Human-readable description
            parameters: Dict of parameter name -> description
        """
        if name in self.tools:
            logger.warning(f"Overwriting existing tool: {name}")

        self.tools[name] = Tool(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters or {},
        )
        logger.info(f"Tool registered: {name}")

    def execute(self, action: str, parameters: Dict[str, Any]) -> str:
        """
        Execute a tool by name with parameters.

        Args:
            action: Tool name to execute
            parameters: Dict of parameters for the tool

        Returns:
            String result from tool execution

        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        if action not in self.tools:
            raise ValueError(f"Unknown action: {action}")

        tool = self.tools[action]
        try:
            logger.info(f"Executing tool: {action} with params: {parameters}")
            result = tool.handler(**parameters)
            logger.info(f"Tool {action} executed successfully")
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {action}: {e}")
            raise

    def get_tools_description(self) -> str:
        """
        Get formatted description of all available tools.
        Useful for system prompts.

        Returns:
            Formatted string describing all tools
        """
        lines = ["Available Tools:"]
        for name, tool in self.tools.items():
            lines.append(f"\n- {name}: {tool.description}")
            if tool.parameters:
                lines.append("  Parameters:")
                for param, desc in tool.parameters.items():
                    lines.append(f"    - {param}: {desc}")
        return "\n".join(lines)

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """
        Get JSON schema of all tools.
        Use this in system prompt to guide LLM toward correct JSON format.

        Returns:
            List of tool schemas
        """
        schemas = []
        for name, tool in self.tools.items():
            schema = {
                "action": name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            schemas.append(schema)
        return schemas

    def _register_default_tools(self):
        """Register built-in tools."""
        self.register(
            "open_app",
            self._open_app,
            "Open an application or URL",
            {
                "app": "Application name or URL (e.g., 'spotify', 'https://example.com')"
            },
        )

        self.register(
            "search_web",
            self._search_web,
            "Search the web for information",
            {"query": "Search query string"},
        )

        self.register(
            "open_file",
            self._open_file,
            "Open a file at specified path",
            {"path": "File path (absolute or relative)"},
        )

        self.register(
            "write_file",
            self._write_file,
            "Write content to a file",
            {
                "path": "File path",
                "content": "Content to write",
            },
        )

        self.register(
            "read_file",
            self._read_file,
            "Read content from a file",
            {"path": "File path"},
        )

    # Tool Implementations

    @staticmethod
    def _open_app(app: str) -> str:
        """Open an application or URL."""
        try:
            # Check if it's a URL
            if app.startswith("http://") or app.startswith("https://"):
                webbrowser.open(app)
                return f"Opened URL: {app}"

            # Try to open as application (macOS/Linux)
            if Path(f"/Applications/{app}.app").exists():
                subprocess.run(
                    ["open", "-a", app],
                    check=True,
                    timeout=5,
                )
                return f"Opened application: {app}"

            # Try generic open command
            subprocess.run(
                ["open", app],
                check=True,
                timeout=5,
            )
            return f"Opened: {app}"

        except Exception as e:
            return f"Failed to open {app}: {str(e)}"

    @staticmethod
    def _search_web(query: str) -> str:
        """Search the web (opens search in browser)."""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Searching web for: {query}"
        except Exception as e:
            return f"Failed to search: {str(e)}"

    @staticmethod
    def _open_file(path: str) -> str:
        """Open a file with default application."""
        try:
            file_path = Path(path).expanduser()
            if not file_path.exists():
                return f"File not found: {path}"

            subprocess.run(
                ["open", str(file_path)],
                check=True,
                timeout=5,
            )
            return f"Opened file: {path}"
        except Exception as e:
            return f"Failed to open file: {str(e)}"

    @staticmethod
    def _read_file(path: str) -> str:
        """Read file content."""
        try:
            file_path = Path(path).expanduser()
            if not file_path.exists():
                return f"File not found: {path}"

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Limit output size
            max_chars = 2000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... (truncated, total: {len(content)} chars)"

            return content
        except Exception as e:
            return f"Failed to read file: {str(e)}"

    @staticmethod
    def _write_file(path: str, content: str) -> str:
        """Write content to file."""
        try:
            file_path = Path(path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"File written successfully: {path}"
        except Exception as e:
            return f"Failed to write file: {str(e)}"
