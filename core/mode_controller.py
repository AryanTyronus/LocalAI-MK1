from typing import Dict
from core.config import get_mode_config


class Mode:
    name = 'base'

    def __init__(self):
        conf = get_mode_config(self.name) or {}
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
    def __init__(self):
        super().__init__()
        # Agents are tool-enabled by design
        self.use_tools = True


class ModeController:
    _modes: Dict[str, Mode] = {
        'chat': ChatMode,
        'coding': CodingMode,
        'research': ResearchMode,
        'agent': AgentMode,
    }

    @classmethod
    def get_mode(cls, name: str) -> Mode:
        name = (name or 'chat').lower()
        mode_cls = cls._modes.get(name, ChatMode)
        return mode_cls()

    @classmethod
    def register_mode(cls, name: str, mode_cls):
        cls._modes[name] = mode_cls
