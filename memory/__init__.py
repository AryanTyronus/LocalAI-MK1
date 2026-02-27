"""
3-Layer Memory Architecture

Exports:
- MemoryManager: Main coordinator for all memory layers
- LongTermMemory: Structured long-term memory
- VectorMemory: Namespaced vector memory
- MemoryStore: Legacy semantic memory (for compatibility)
"""

from memory.memory_manager import MemoryManager, StructuredMemory
from memory.long_term_memory import LongTermMemory
from memory.vector_memory import VectorMemory
from memory.semantic_memory import SemanticMemory
from memory.summarizer import ConversationSummarizer
from memory.memory_store import MemoryStore

__all__ = [
    'MemoryManager',
    'StructuredMemory',
    'LongTermMemory',
    'VectorMemory',
    'SemanticMemory',
    'ConversationSummarizer',
    'MemoryStore'  # Legacy
]
