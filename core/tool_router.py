"""
Tool Router - Preparation for future tool execution.

This module provides a stub for detecting tool intent from user messages.
It prepares the system for future OS-level tool integration without
actually implementing any dangerous operations.

Current status: STUB - Returns structured tool call format
Future: Will integrate with actual tool execution system
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.logger import logger


class ToolIntent(Enum):
    """Enum for detected tool intents."""
    NONE = "none"
    OPEN_APP = "open_app"
    SEARCH_WEB = "search_web"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    RUN_CODE = "run_code"
    CUSTOM = "custom"


@dataclass
class ToolCall:
    """Structured tool call representation."""
    intent: ToolIntent
    tool_name: str
    parameters: Dict
    confidence: float
    requires_confirmation: bool = True


class ToolRouter:
    """
    Router for detecting tool intent from user messages.
    
    This is a STUB implementation for future tool integration.
    Currently provides:
    - Intent detection patterns
    - Structured tool call format
    - Extension points for future tools
    
    NO actual OS commands are executed.
    """
    
    # Patterns for intent detection
    INTENT_PATTERNS = {
        ToolIntent.OPEN_APP: [
            r'open\s+(?:the\s+)?(.+)',
            r'launch\s+(?:the\s+)?(.+)',
            r'start\s+(?:the\s+)?(.+)',
        ],
        ToolIntent.SEARCH_WEB: [
            r'search\s+(?:for\s+)?(.+)',
            r'look\s+up\s+(.+)',
            r'find\s+(?:information\s+about\s+)?(.+)',
        ],
        ToolIntent.READ_FILE: [
            r'read\s+(?:the\s+)?file\s+(.+)',
            r'open\s+file\s+(.+)',
            r'show\s+(?:me\s+)?(?:the\s+)?content\s+of\s+(.+)',
        ],
        ToolIntent.WRITE_FILE: [
            r'write\s+(?:to\s+)?(?:the\s+)?file\s+(.+)',
            r'save\s+(?:to\s+)?(?:the\s+)?file\s+(.+)',
            r'create\s+(?:a\s+)?file\s+(?:called\s+)?(.+)',
        ],
        ToolIntent.RUN_CODE: [
            r'run\s+(?:the\s+)?code',
            r'execute\s+(?:the\s+)?code',
            r'run\s+(?:python|javascript|js)\s+code',
        ],
    }
    
    # Tool definitions (for future extension)
    REGISTERED_TOOLS = {
        'open_app': {
            'description': 'Open an application',
            'parameters': {'app_name': 'string'},
            'requires_confirmation': True,
        },
        'search_web': {
            'description': 'Search the web',
            'parameters': {'query': 'string'},
            'requires_confirmation': False,
        },
        'read_file': {
            'description': 'Read a file from disk',
            'parameters': {'filepath': 'string'},
            'requires_confirmation': True,
        },
        'write_file': {
            'description': 'Write content to a file',
            'parameters': {'filepath': 'string', 'content': 'string'},
            'requires_confirmation': True,
        },
        'run_code': {
            'description': 'Execute code',
            'parameters': {'language': 'string', 'code': 'string'},
            'requires_confirmation': True,
        },
    }
    
    def __init__(self):
        """Initialize the tool router."""
        self._enabled = False  # Disabled by default - just a stub
        logger.info("ToolRouter initialized (STUB - no actual execution)")
    
    @property
    def is_enabled(self) -> bool:
        """Check if tool routing is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable tool routing (for future use)."""
        self._enabled = True
        logger.info("ToolRouter enabled")
    
    def disable(self) -> None:
        """Disable tool routing."""
        self._enabled = False
        logger.info("ToolRouter disabled")
    
    def detect_intent(self, message: str) -> ToolCall:
        """
        Detect tool intent from user message.
        
        Args:
            message: User message
            
        Returns:
            ToolCall with detected intent (or NONE if no match)
        """
        if not self._enabled:
            return ToolCall(
                intent=ToolIntent.NONE,
                tool_name="",
                parameters={},
                confidence=0.0,
                requires_confirmation=False
            )
        
        message_lower = message.lower()
        
        # Check each intent pattern
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    # Extract parameters from match
                    params = self._extract_parameters(intent, match)
                    
                    return ToolCall(
                        intent=intent,
                        tool_name=self._intent_to_tool_name(intent),
                        parameters=params,
                        confidence=0.8,  # Default confidence
                        requires_confirmation=self._requires_confirmation(intent)
                    )
        
        # No intent detected
        return ToolCall(
            intent=ToolIntent.NONE,
            tool_name="",
            parameters={},
            confidence=0.0,
            requires_confirmation=False
        )
    
    def _extract_parameters(self, intent: ToolIntent, match: re.Match) -> Dict:
        """Extract parameters from regex match."""
        params = {}
        
        if intent == ToolIntent.OPEN_APP:
            if match.group(1):
                params['app_name'] = match.group(1).strip()
        
        elif intent == ToolIntent.SEARCH_WEB:
            if match.group(1):
                params['query'] = match.group(1).strip()
        
        elif intent == ToolIntent.READ_FILE:
            if match.group(1):
                params['filepath'] = match.group(1).strip()
        
        elif intent == ToolIntent.WRITE_FILE:
            if match.group(1):
                params['filepath'] = match.group(1).strip()
        
        return params
    
    def _intent_to_tool_name(self, intent: ToolIntent) -> str:
        """Convert intent to tool name."""
        mapping = {
            ToolIntent.OPEN_APP: "open_app",
            ToolIntent.SEARCH_WEB: "search_web",
            ToolIntent.READ_FILE: "read_file",
            ToolIntent.WRITE_FILE: "write_file",
            ToolIntent.RUN_CODE: "run_code",
        }
        return mapping.get(intent, "")
    
    def _requires_confirmation(self, intent: ToolIntent) -> bool:
        """Check if intent requires confirmation."""
        dangerous_intents = {
            ToolIntent.OPEN_APP,
            ToolIntent.WRITE_FILE,
            ToolIntent.RUN_CODE,
        }
        return intent in dangerous_intents
    
    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Format tool call as JSON string.
        
        This is the output format expected by the pipeline.
        
        Args:
            tool_call: ToolCall to format
            
        Returns:
            JSON string representation
        """
        if tool_call.intent == ToolIntent.NONE:
            return ""
        
        import json
        return json.dumps({
            'intent': tool_call.intent.value,
            'tool': tool_call.tool_name,
            'parameters': tool_call.parameters,
            'confidence': tool_call.confidence,
            'requires_confirmation': tool_call.requires_confirmation
        })
    
    def should_route_to_tools(self, message: str) -> bool:
        """
        Check if message should be routed to tool execution.
        
        Args:
            message: User message
            
        Returns:
            True if tool execution should be attempted
        """
        if not self._enabled:
            return False
        
        tool_call = self.detect_intent(message)
        return tool_call.intent != ToolIntent.NONE
    
    # ============================================
    # Extension Points for Future Tools
    # ============================================
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict,
        requires_confirmation: bool = True
    ) -> None:
        """
        Register a new tool for routing.
        
        This allows future extension without modifying core logic.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
            requires_confirmation: Whether tool needs confirmation
        """
        self.REGISTERED_TOOLS[name] = {
            'description': description,
            'parameters': parameters,
            'requires_confirmation': requires_confirmation
        }
        logger.info(f"ToolRouter: Registered tool '{name}'")
    
    def add_intent_pattern(
        self,
        intent: ToolIntent,
        pattern: str
    ) -> None:
        """
        Add a new intent pattern.
        
        Allows extending detection without code changes.
        
        Args:
            intent: ToolIntent enum value
            pattern: Regex pattern
        """
        if intent not in self.INTENT_PATTERNS:
            self.INTENT_PATTERNS[intent] = []
        
        self.INTENT_PATTERNS[intent].append(pattern)
        logger.info(f"ToolRouter: Added pattern for {intent.value}")
    
    def get_available_tools(self) -> List[Dict]:
        """
        Get list of available tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                'name': name,
                **definition
            }
            for name, definition in self.REGISTERED_TOOLS.items()
        ]


# Global instance for easy access
_tool_router = None


def get_tool_router() -> ToolRouter:
    """Get the global ToolRouter instance."""
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router

