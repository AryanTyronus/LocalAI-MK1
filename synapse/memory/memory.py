"""Conversation Memory - Lightweight history management for SYNAPSE."""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action: Optional[str] = None  # Tool action name if any
    thinking: Optional[str] = None  # LLM reasoning/thinking


class ConversationMemory:
    """
    Lightweight conversation memory system.
    Stores last N interactions for context window.
    """

    def __init__(
        self,
        max_messages: int = 20,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum messages to keep in memory
            persist_path: Path to persist memory to disk (optional)
        """
        self.max_messages = max_messages
        self.persist_path = persist_path
        self.messages: List[Message] = []

        if persist_path:
            self.load()

    def add_message(
        self,
        role: str,
        content: str,
        action: Optional[str] = None,
        thinking: Optional[str] = None,
    ):
        """
        Add a message to memory.

        Args:
            role: "user", "assistant", or "system"
            content: Message content
            action: Tool action if executed
            thinking: LLM thinking/reasoning
        """
        message = Message(
            role=role,
            content=content,
            action=action,
            thinking=thinking,
        )
        self.messages.append(message)

        # Trim to max size (keep most recent)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        logger.debug(f"Message added: {role} ({len(message.content)} chars)")

        if self.persist_path:
            self.save()

    def get_messages(self, include_thinking: bool = False) -> List[Dict[str, str]]:
        """
        Get messages in OpenAI-compatible format.

        Args:
            include_thinking: Include thinking field if available

        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        for msg in self.messages:
            entry = {
                "role": msg.role,
                "content": msg.content,
            }
            if include_thinking and msg.thinking:
                entry["thinking"] = msg.thinking
            messages.append(entry)
        return messages

    def get_summary(self) -> Dict[str, Any]:
        """
        Get memory summary statistics.

        Returns:
            Dict with counts and recent interactions
        """
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
            "tools_used": [m.action for m in self.messages if m.action],
            "recent": [
                {
                    "role": m.role,
                    "preview": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    "timestamp": m.timestamp,
                }
                for m in self.messages[-3:]
            ],
        }

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
        if self.persist_path:
            Path(self.persist_path).unlink(missing_ok=True)
        logger.info("Memory cleared")

    def save(self):
        """Save conversation to disk."""
        try:
            if not self.persist_path:
                return

            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "messages": [asdict(m) for m in self.messages],
                "saved_at": datetime.now().isoformat(),
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Memory saved to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def load(self):
        """Load conversation from disk."""
        try:
            if not self.persist_path or not Path(self.persist_path).exists():
                return

            with open(self.persist_path, "r") as f:
                data = json.load(f)

            self.messages = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp"),
                    action=m.get("action"),
                    thinking=m.get("thinking"),
                )
                for m in data.get("messages", [])
            ]

            logger.info(f"Loaded {len(self.messages)} messages from disk")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            self.messages = []
