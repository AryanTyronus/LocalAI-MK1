import os
import yaml

MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ============================================
# FILE PATHS
# ============================================
MEMORY_FILE = os.path.join(BASE_DIR, "structured_memory.json")
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge")

# ============================================
# LEGACY SETTINGS (for backward compatibility)
# ============================================
MAX_HISTORY_TURNS = 5
MEMORY_TOP_K = 3
DOC_TOP_K = 2

STUDY_KEYWORDS = [
    "physics", "math", "jee", "derive",
    "equation", "numerical", "solve",
    "formula", "calculate", "explain"
]

# ============================================
# LOAD YAML CONFIG FOR 4-LAYER MEMORY
# ============================================
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")

def load_config():
    """Load configuration from YAML file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Load memory configuration
_config = load_config()
DEBUG_CONFIG = _config.get('debug', {})
MEMORY_CONFIG = _config.get('memory', {})
SUMMARIZATION_CONFIG = _config.get('summarization', {})
SEARCH_CONFIG = _config.get('search', {})

# Memory layer configurations
SHORT_TERM_CONFIG = MEMORY_CONFIG.get('short_term', {'max_messages': 10, 'enabled': True})
ROLLING_SUMMARY_CONFIG = MEMORY_CONFIG.get('rolling_summary', {
    'enabled': True,
    'trigger_threshold': 15,
    'summary_size': 3
})
SEMANTIC_CONFIG = MEMORY_CONFIG.get('semantic', {
    'enabled': True,
    'similarity_threshold': 0.5,
    'max_results': 3
})
STRUCTURED_CONFIG = MEMORY_CONFIG.get('structured', {'enabled': True, 'file': MEMORY_FILE})

# Thresholds
MEMORY_THRESHOLDS = MEMORY_CONFIG.get('thresholds', {})
MAX_TOTAL_MEMORY_MB = MEMORY_THRESHOLDS.get('max_total_memory_mb', 500)
CLEANUP_TRIGGER = MEMORY_THRESHOLDS.get('cleanup_trigger', 0.85)

# Debug & Error Handling
DEBUG_MODE = DEBUG_CONFIG.get('enabled', False)
PRINT_STACK_TRACES = DEBUG_CONFIG.get('print_stack_traces', True)

# Summarization
SUMMARIZATION_MODEL_TYPE = SUMMARIZATION_CONFIG.get('model_type', 'extractive')
KEY_PHRASES_COUNT = SUMMARIZATION_CONFIG.get('key_phrases_count', 5)
PRESERVE_UNRESOLVED = SUMMARIZATION_CONFIG.get('preserve_unresolved', True)

# Search
DEFAULT_TOP_K = SEARCH_CONFIG.get('default_top_k', 3)
INCLUDE_SUMMARIES_IN_SEARCH = SEARCH_CONFIG.get('include_summaries', True)
INCLUDE_SEMANTIC_IN_SEARCH = SEARCH_CONFIG.get('include_semantic', True)

# Modes
MODE_CONFIG = _config.get('modes', {})

def get_mode_config(mode: str) -> dict:
    return MODE_CONFIG.get(mode, {})
