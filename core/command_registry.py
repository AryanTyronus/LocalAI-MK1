"""
Slash-command registry for metadata-driven command discovery and execution.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.logger import logger


@dataclass
class SlashCommand:
    """Metadata and handler for a slash command."""

    name: str
    description: str
    category: str
    usage: Optional[str]
    handler: Callable[[Dict[str, Any], str], Dict[str, Any]]
    store_in_memory: bool = False


class CommandRegistry:
    """Central registry for slash commands."""

    _commands: Dict[str, SlashCommand] = {}

    @classmethod
    def register_command(
        cls,
        name: str,
        description: str,
        category: str,
        handler: Callable[[Dict[str, Any], str], Dict[str, Any]],
        usage: Optional[str] = None,
        store_in_memory: bool = False,
    ) -> None:
        normalized = cls.normalize_name(name)
        cls._commands[normalized] = SlashCommand(
            name=normalized,
            description=description,
            category=(category or "general").strip().lower(),
            usage=usage,
            handler=handler,
            store_in_memory=store_in_memory,
        )
        logger.info(f"Registered command: {normalized}")

    @classmethod
    def get_command(cls, name: str) -> Optional[SlashCommand]:
        return cls._commands.get(cls.normalize_name(name))

    @classmethod
    def list_commands(cls) -> List[SlashCommand]:
        return sorted(cls._commands.values(), key=lambda cmd: cmd.name)

    @staticmethod
    def normalize_name(name: str) -> str:
        normalized = (name or "").strip().lower()
        if not normalized:
            return "/"
        return normalized if normalized.startswith("/") else f"/{normalized}"
