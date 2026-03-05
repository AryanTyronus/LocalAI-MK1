"""
Capability executor with argument sanitization and path restrictions.
"""

from __future__ import annotations

import os
from typing import Dict

from core.config import BASE_DIR
from core.logger import logger
from capabilities.registry import CapabilityRegistry


class CapabilityExecutor:
    def __init__(self, tool_registry):
        self._tool_registry = tool_registry
        self._registry = CapabilityRegistry()
        self._base_dir = os.path.abspath(BASE_DIR)

    def _sanitize(self, capability_name: str, params: Dict) -> Dict:
        safe = dict(params or {})

        # Restrict file paths to project root.
        for key in ("filepath", "path", "file"):
            if key in safe and isinstance(safe[key], str):
                requested = os.path.abspath(os.path.join(self._base_dir, safe[key]))
                if not requested.startswith(self._base_dir):
                    raise ValueError("Path outside workspace is not allowed")
                safe[key] = requested

        # Prevent shell-ish payloads in generic string params.
        for key, value in list(safe.items()):
            if isinstance(value, str):
                lowered = value.lower()
                blocked = ("&&", "||", ";", "`", "$(", "rm -rf", "shutdown", "reboot")
                if any(token in lowered for token in blocked):
                    raise ValueError(f"Unsafe argument rejected: {key}")

        return safe

    def execute(self, tool_name: str, params: Dict, require_confirmation: bool = False) -> Dict:
        capability = self._registry.resolve_from_tool(tool_name)
        if capability is None:
            logger.warning(f"No capability mapping for tool '{tool_name}', executing with base registry")
            return self._tool_registry.execute_tool(tool_name, dict(params or {}), require_confirmation=require_confirmation)

        safe_params = self._sanitize(capability.name, params or {})
        return self._tool_registry.execute_tool(
            capability.tool_name,
            safe_params,
            require_confirmation=require_confirmation or capability.dangerous,
        )

