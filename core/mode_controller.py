"""
ModeController module - Handles different operational modes.
Accepts Config via constructor for strict config usage.
"""

from typing import Dict
from core.config import Config


class Mode:
    """Base mode class with configuration from config."""
    
    name = 'base'
    
    def __init__(self, config: Config = None):
        """
        Initialize Mode with configuration.
        
        Args:
            config: Config instance for settings
        """
        self._config = config or Config()
        conf = self._config.get_mode_config(self.name) or {}
        self.system_prompt = conf.get('system_prompt', '')
        self.temperature = float(conf.get('temperature', 0.7))
        self.max_tokens = int(conf.get('max_tokens', 512))
        self.retrieve_documents = bool(conf.get('retrieve_documents', True))
        self.use_tools = bool(conf.get('use_tools', False))
        self.memory_profile = conf.get('memory_profile', '')


class ChatMode(Mode):
    name = 'chat'


class CodingMode(Mode):
    name = 'coding'


class ResearchMode(Mode):
    name = 'research'


class AgentMode(Mode):
    name = 'agent'
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        # Agents are tool-enabled by design
        self.use_tools = True


class ModeController:
    """
    Controller for managing different operational modes.
    Modes determine system prompt, temperature, document retrieval, and memory profile.
    """
    
    _modes: Dict[str, type] = {
        'chat': ChatMode,
        'coding': CodingMode,
        'research': ResearchMode,
        'agent': AgentMode,
    }
    
    _config: Config = None
    
    @classmethod
    def configure(cls, config: Config):
        """
        Configure the ModeController with a Config instance.
        
        Args:
            config: Config instance for settings
        """
        cls._config = config
    
    @classmethod
    def _get_config(cls) -> Config:
        """Get the configured Config instance."""
        return cls._config or Config()
    
    @classmethod
    def get_mode(cls, name: str) -> Mode:
        """
        Get a mode instance by name.
        
        Args:
            name: Mode name (chat, coding, research, agent)
            
        Returns:
            Mode instance with appropriate configuration
        """
        name = (name or 'chat').lower()
        mode_cls = cls._modes.get(name, ChatMode)
        config = cls._get_config()
        return mode_cls(config)
    
    @classmethod
    def register_mode(cls, name: str, mode_cls: type):
        """
        Register a new mode.
        
        Args:
            name: Mode name
            mode_cls: Mode class
        """
        cls._modes[name] = mode_cls

