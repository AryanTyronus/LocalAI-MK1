"""
Config module - Centralized configuration management.
All settings must come from config.yaml - no hardcoded values.
"""

import os
import yaml
from typing import Any, Dict, Optional


class Config:
    """
    Centralized configuration class.
    All thresholds and settings must be accessed through this class.
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, "config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            self._config: Dict = yaml.safe_load(f)
        
        self._initialized = True
        
        # Model configuration
        self._model_config = self._config.get('model', {})
        
        # Memory configuration
        self._memory_config = self._config.get('memory', {})
        
        # Debug configuration
        self._debug_config = self._config.get('debug', {})
        
        # Summarization configuration
        self._summarization_config = self._config.get('summarization', {})
        
        # Search configuration
        self._search_config = self._config.get('search', {})
        
        # Modes configuration
        self._modes_config = self._config.get('modes', {})
    
    # ===================
    # Model Settings
    # ===================
    
    @property
    def model_name(self) -> str:
        """Get the default model name from config."""
        return self._model_config.get('default', 'qwen-7b')
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens from config."""
        return int(self._model_config.get('max_tokens', 2048))
    
    # ===================
    # Memory Settings
    # ===================
    
    @property
    def short_term_max_messages(self) -> int:
        """Get short-term memory max messages."""
        return int(self._memory_config.get('short_term', {}).get('max_messages', 10))
    
    @property
    def short_term_enabled(self) -> bool:
        """Get short-term memory enabled status."""
        return bool(self._memory_config.get('short_term', {}).get('enabled', True))
    
    @property
    def rolling_summary_enabled(self) -> bool:
        """Get rolling summary enabled status."""
        return bool(self._memory_config.get('rolling_summary', {}).get('enabled', True))
    
    @property
    def rolling_summary_trigger_threshold(self) -> int:
        """Get rolling summary trigger threshold."""
        return int(self._memory_config.get('rolling_summary', {}).get('trigger_threshold', 15))
    
    @property
    def rolling_summary_size(self) -> int:
        """Get rolling summary size."""
        return int(self._memory_config.get('rolling_summary', {}).get('summary_size', 3))
    
    @property
    def semantic_enabled(self) -> bool:
        """Get semantic memory enabled status."""
        return bool(self._memory_config.get('semantic', {}).get('enabled', True))
    
    @property
    def semantic_embedding_model(self) -> str:
        """Get semantic memory embedding model."""
        return self._memory_config.get('semantic', {}).get('embedding_model', 'all-MiniLM-L6-v2')
    
    @property
    def semantic_similarity_threshold(self) -> float:
        """Get semantic memory similarity threshold."""
        return float(self._memory_config.get('semantic', {}).get('similarity_threshold', 0.5))
    
    @property
    def semantic_max_results(self) -> int:
        """Get semantic memory max results."""
        return int(self._memory_config.get('semantic', {}).get('max_results', 3))
    
    @property
    def semantic_persist_interval(self) -> int:
        """Get semantic memory persist interval."""
        return int(self._memory_config.get('semantic', {}).get('persist_interval', 10))
    
    @property
    def structured_enabled(self) -> bool:
        """Get structured memory enabled status."""
        return bool(self._memory_config.get('structured', {}).get('enabled', True))
    
    @property
    def structured_file(self) -> str:
        """Get structured memory file path."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        return self._memory_config.get('structured', {}).get('file', os.path.join(base_dir, 'structured_memory.json'))
    
    @property
    def structured_auto_extract_profile(self) -> bool:
        """Get structured memory auto extract profile."""
        return bool(self._memory_config.get('structured', {}).get('auto_extract_profile', True))
    
    @property
    def memory_thresholds_max_total_mb(self) -> int:
        """Get memory thresholds max total MB."""
        return int(self._memory_config.get('thresholds', {}).get('max_total_memory_mb', 500))
    
    @property
    def memory_thresholds_cleanup_trigger(self) -> float:
        """Get memory thresholds cleanup trigger."""
        return float(self._memory_config.get('thresholds', {}).get('cleanup_trigger', 0.85))
    
    # ===================
    # Debug Settings
    # ===================
    
    @property
    def debug_enabled(self) -> bool:
        """Get debug enabled status."""
        return bool(self._debug_config.get('enabled', False))
    
    @property
    def debug_print_stack_traces(self) -> bool:
        """Get debug print stack traces status."""
        return bool(self._debug_config.get('print_stack_traces', True))
    
    # ===================
    # Summarization Settings
    # ===================
    
    @property
    def summarization_model_type(self) -> str:
        """Get summarization model type."""
        return self._summarization_config.get('model_type', 'extractive')
    
    @property
    def summarization_key_phrases_count(self) -> int:
        """Get summarization key phrases count."""
        return int(self._summarization_config.get('key_phrases_count', 5))
    
    @property
    def summarization_preserve_unresolved(self) -> bool:
        """Get summarization preserve unresolved."""
        return bool(self._summarization_config.get('preserve_unresolved', True))
    
    # ===================
    # Search Settings
    # ===================
    
    @property
    def search_default_top_k(self) -> int:
        """Get search default top k."""
        return int(self._search_config.get('default_top_k', 3))
    
    @property
    def search_include_summaries(self) -> bool:
        """Get search include summaries."""
        return bool(self._search_config.get('include_summaries', True))
    
    @property
    def search_include_semantic(self) -> bool:
        """Get search include semantic."""
        return bool(self._search_config.get('include_semantic', True))
    
    # ===================
    # Mode Settings
    # ===================
    
    def get_mode_config(self, mode: str) -> Dict:
        """Get configuration for a specific mode."""
        return self._modes_config.get(mode, {})
    
    @property
    def modes(self) -> Dict:
        """Get all mode configurations."""
        return self._modes_config
    
    # ===================
    # General Settings
    # ===================
    
    @property
    def knowledge_folder(self) -> str:
        """Get knowledge folder path."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        return self._config.get('knowledge_folder', os.path.join(base_dir, 'knowledge'))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a generic config value."""
        return self._config.get(key, default)


# ============================================
# Legacy module-level constants (backward compatibility)
# ============================================

# Load the singleton config
_config = Config()

MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# File paths
MEMORY_FILE = _config.structured_file
KNOWLEDGE_FOLDER = _config.knowledge_folder

# Legacy settings (for backward compatibility)
MAX_HISTORY_TURNS = 5
MEMORY_TOP_K = 3
DOC_TOP_K = 2

STUDY_KEYWORDS = [
    "physics", "math", "jee", "derive",
    "equation", "numerical", "solve",
    "formula", "calculate", "explain"
]

# Memory configurations (from Config class)
SHORT_TERM_CONFIG = {
    'max_messages': _config.short_term_max_messages,
    'enabled': _config.short_term_enabled
}

ROLLING_SUMMARY_CONFIG = {
    'enabled': _config.rolling_summary_enabled,
    'trigger_threshold': _config.rolling_summary_trigger_threshold,
    'summary_size': _config.rolling_summary_size
}

SEMANTIC_CONFIG = {
    'enabled': _config.semantic_enabled,
    'embedding_model': _config.semantic_embedding_model,
    'similarity_threshold': _config.semantic_similarity_threshold,
    'max_results': _config.semantic_max_results
}

STRUCTURED_CONFIG = {
    'enabled': _config.structured_enabled,
    'file': _config.structured_file
}

# Thresholds
MEMORY_THRESHOLDS = {
    'max_total_memory_mb': _config.memory_thresholds_max_total_mb,
    'cleanup_trigger': _config.memory_thresholds_cleanup_trigger
}

MAX_TOTAL_MEMORY_MB = _config.memory_thresholds_max_total_mb
CLEANUP_TRIGGER = _config.memory_thresholds_cleanup_trigger

# Debug & Error Handling
DEBUG_MODE = _config.debug_enabled
PRINT_STACK_TRACES = _config.debug_print_stack_traces

# Summarization
SUMMARIZATION_MODEL_TYPE = _config.summarization_model_type
KEY_PHRASES_COUNT = _config.summarization_key_phrases_count
PRESERVE_UNRESOLVED = _config.summarization_preserve_unresolved

# Search
DEFAULT_TOP_K = _config.search_default_top_k
INCLUDE_SUMMARIES_IN_SEARCH = _config.search_include_summaries
INCLUDE_SEMANTIC_IN_SEARCH = _config.search_include_semantic

# Modes
MODE_CONFIG = _config.modes


def get_mode_config(mode: str) -> dict:
    """Get mode configuration (legacy function)."""
    return _config.get_mode_config(mode)

