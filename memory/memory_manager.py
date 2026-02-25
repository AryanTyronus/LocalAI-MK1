"""
4-Layer Memory Architecture Manager

Coordinates four memory layers:
1. Short-term memory (active conversation)
2. Rolling summary memory (compressed history)
3. Semantic memory (vector-indexed facts)
4. Structured persistent memory (user profile)
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import deque

from core.config import (
    MEMORY_FILE,
    SHORT_TERM_CONFIG,
    ROLLING_SUMMARY_CONFIG,
    SEMANTIC_CONFIG,
    STRUCTURED_CONFIG,
    INCLUDE_SUMMARIES_IN_SEARCH,
    INCLUDE_SEMANTIC_IN_SEARCH,
    DEFAULT_TOP_K
)
from core.logger import logger
from core.model_manager import ModelManager
from memory.summarizer import ConversationSummarizer
from memory.semantic_memory import SemanticMemory


class MemoryManager:
    """
    Unified memory manager coordinating all 4 memory layers.
    Provides a clean interface for the rest of the application.
    """

    def __init__(self):
        """Initialize the 4-layer memory system."""
        self.model_manager = ModelManager.get_instance()

        # Layer 1: Short-term memory
        self.short_term_max = SHORT_TERM_CONFIG.get('max_messages', 10)
        self.short_term = deque(maxlen=self.short_term_max)

        # Layer 2: Rolling summary memory
        self.rolling_summaries = []
        self.summary_trigger = ROLLING_SUMMARY_CONFIG.get('trigger_threshold', 15)
        self.summarizer = ConversationSummarizer()

        # Layer 3: Semantic memory (vector-indexed)
        self.semantic = SemanticMemory(self.model_manager)

        # Layer 4: Structured persistent memory
        self.structured = StructuredMemory(MEMORY_FILE)

        logger.info("4-Layer Memory Manager initialized")

    # ================================================
    # LAYER 1: SHORT-TERM MEMORY (Active window)
    # ================================================

    def add_short_term_message(self, role: str, content: str) -> None:
        """
        Add a message to short-term memory.
        Automatically discards oldest when limit reached.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.short_term.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Added {role} message to short-term (total: {len(self.short_term)})")

    def get_short_term_context(self) -> str:
        """
        Get formatted short-term memory for prompt inclusion.

        Returns:
            Formatted conversation context
        """
        if not self.short_term:
            return ""

        lines = []
        for msg in self.short_term:
            role = msg['role'].capitalize()
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def get_short_term_messages(self) -> List[Dict]:
        """Get raw short-term messages."""
        return list(self.short_term)

    # ================================================
    # LAYER 2: ROLLING SUMMARY MEMORY (Compressed history)
    # ================================================

    def maybe_create_summary(self) -> bool:
        """
        Check if we should create a rolling summary.
        Triggered when total messages exceed threshold.

        Returns:
            True if summary was created
        """
        total_messages = len(self.short_term)

        if total_messages >= self.summary_trigger:
            self._create_rolling_summary()
            return True

        return False

    def _create_rolling_summary(self) -> None:
        """Create a summary from short-term memory and add to rolling memory."""
        if not self.short_term:
            return

        messages = list(self.short_term)

        # Generate summary
        summary = self.summarizer.summarize_messages(messages)
        self.rolling_summaries.append(summary)

        # Keep only recent summaries
        max_summaries = ROLLING_SUMMARY_CONFIG.get('summary_size', 3)
        if len(self.rolling_summaries) > max_summaries:
            self.rolling_summaries = self.rolling_summaries[-max_summaries:]

        logger.info(f"Created rolling summary from {len(messages)} messages")

    def get_rolling_summary_context(self) -> str:
        """
        Get formatted rolling summary memory for prompt inclusion.

        Returns:
            Formatted summary context
        """
        if not self.rolling_summaries:
            return ""

        lines = ["=== Conversation History Summary ==="]
        for i, summary in enumerate(self.rolling_summaries[-2:], 1):  # Last 2 summaries
            lines.append(f"\nSummary {i}: {summary['summary_text']}")

        return "\n".join(lines)

    # ================================================
    # LAYER 3: SEMANTIC MEMORY (Vector-indexed facts)
    # ================================================

    def add_semantic_memory(self, text: str, metadata: Dict = None) -> None:
        """
        Add important fact to semantic memory with embeddings.

        Args:
            text: The memory text
            metadata: Optional metadata (source, type, etc.)
        """
        self.semantic.add_memory(text, metadata)
        logger.debug(f"Added to semantic memory: {text[:50]}...")

    def search_semantic_memory(self, query: str, top_k: int = None) -> List[str]:
        """
        Search semantic memory for relevant facts.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memory entries
        """
        if not SEMANTIC_CONFIG.get('enabled', True):
            return []

        top_k = top_k or DEFAULT_TOP_K
        return self.semantic.search(query, k=top_k)

    def get_semantic_context(self, query: str) -> str:
        """
        Get formatted semantic memory context for a query.

        Args:
            query: Current user query

        Returns:
            Formatted semantic memory results
        """
        if not INCLUDE_SEMANTIC_IN_SEARCH:
            return ""

        results = self.search_semantic_memory(query)

        if not results:
            return ""

        return "=== Relevant Memories ===\n" + "\n".join(f"- {r}" for r in results)

    # ================================================
    # LAYER 4: STRUCTURED MEMORY (User profile/state)
    # ================================================

    def update_structured_memory(self, key: str, value: any) -> None:
        """
        Update a value in structured memory.

        Args:
            key: Memory key (e.g., 'user.name', 'user.age')
            value: Value to store
        """
        self.structured.set(key, value)

    def get_structured_memory(self, key: str = None) -> any:
        """
        Retrieve structured memory value.

        Args:
            key: Memory key (or None to get all)

        Returns:
            Memory value or entire memory dict
        """
        if key is None:
            return self.structured.get_all()
        return self.structured.get(key)

    def extract_profile_facts(self, message: str) -> None:
        """
        Auto-extract user profile information from message.
        Looks for patterns like "my name is X", "I'm Y years old", etc.

        Args:
            message: User message to parse
        """
        message_lower = message.lower()

        # Name extraction
        if "my name is" in message_lower:
            name = message.split("my name is")[-1].split(".")[0].strip()
            self.update_structured_memory("user.name", name)

        # Age extraction
        import re
        age_match = re.search(r"i'm?\s+(\d+)\s+years? old", message_lower)
        if age_match:
            age = int(age_match.group(1))
            self.update_structured_memory("user.age", age)

        # Birth year extraction
        year_match = re.search(r"born in?\s+(\d{4})", message_lower)
        if year_match:
            year = int(year_match.group(1))
            self.update_structured_memory("user.birth_year", year)

    # ================================================
    # MULTI-LAYER CONTEXT ASSEMBLY
    # ================================================

    def build_full_context(self, current_query: str) -> Dict[str, str]:
        """
        Build comprehensive context from all memory layers.

        Args:
            current_query: Current user query

        Returns:
            Dict with context from each layer
        """
        context = {
            'short_term': self.get_short_term_context(),
            'rolling_summary': self.get_rolling_summary_context() if INCLUDE_SUMMARIES_IN_SEARCH else "",
            'semantic': self.get_semantic_context(current_query),
            'structured': self._format_structured_context()
        }

        return context

    def _format_structured_context(self) -> str:
        """Format structured memory for inclusion in prompts."""
        structured = self.get_structured_memory()
        # Make the formatter robust to either a dict (full structured memory)
        # or a list (e.g. when callers accidentally return only a subsection).
        if not structured:
            return ""

        # If it's a dict, format the known structured fields.
        if isinstance(structured, dict):
            # If dict is empty or all falsy, nothing to include.
            if not any(structured.values()):
                return ""

            lines = ["=== User Profile ==="]

            user_profile = structured.get('user', {})
            if user_profile:
                if user_profile.get('name'):
                    lines.append(f"Name: {user_profile['name']}")
                if user_profile.get('age'):
                    lines.append(f"Age: {user_profile['age']}")
                if user_profile.get('birth_year'):
                    lines.append(f"Birth Year: {user_profile['birth_year']}")

            preferences = structured.get('preferences', {})
            if preferences:
                lines.append("\nPreferences:")
                for key, value in preferences.items():
                    lines.append(f"  {key}: {value}")

            goals = structured.get('goals', [])
            if goals:
                lines.append("\nGoals:")
                for goal in goals:
                    lines.append(f"  - {goal}")

            return "\n".join(lines) if len(lines) > 1 else ""

        # If it's a list, assume it's a simple list of goals or items and display them.
        if isinstance(structured, list):
            if not structured:
                return ""

            lines = ["=== User Profile ==="]
            # If list of strings, treat as goals; otherwise, stringify entries.
            if all(isinstance(x, str) for x in structured):
                lines.append("\nGoals:")
                for goal in structured:
                    lines.append(f"  - {goal}")
            else:
                for item in structured:
                    lines.append(f"- {item}")

            return "\n".join(lines) if len(lines) > 1 else ""

        # Fallback: stringify unexpected types.
        try:
            return str(structured)
        except Exception:
            return ""

    # ================================================
    # PERSISTENT STORAGE
    # ================================================

    def save_all(self) -> None:
        """Save all persistent memory to disk."""
        self.structured.save()
        self.semantic.save()
        logger.info("All memory layers saved")

    def clear_short_term(self) -> None:
        """Clear short-term memory (for new conversations)."""
        self.short_term.clear()
        logger.info("Short-term memory cleared")


class StructuredMemory:
    """
    Layer 4: Structured persistent memory.
    Stores user profile, preferences, goals, and system state as JSON.
    """

    def __init__(self, filepath: str):
        """
        Initialize structured memory.

        Args:
            filepath: Path to JSON storage file
        """
        self.filepath = filepath
        self.data = self._load() or {
            'user': {},
            'preferences': {},
            'goals': [],
            'system_state': {},
            'created_at': datetime.now().isoformat()
        }

    def _load(self) -> Optional[Dict]:
        """Load structured memory from file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    # Expect a dict for structured memory; if file contains other
                    # formats (e.g., a list from older semantic dumps), ignore it
                    if isinstance(data, dict):
                        return data
                    else:
                        logger.warning("Structured memory file has unexpected format; ignoring and starting fresh.")
                        return None
            except Exception as e:
                logger.warning(f"Failed to load structured memory: {e}")
        return None

    def set(self, key: str, value: any) -> None:
        """
        Set a value using dot notation (e.g., 'user.name').

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        current = self.data

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        self.save()

    def get(self, key: str) -> Optional[any]:
        """
        Get a value using dot notation.

        Args:
            key: Dot-separated key path

        Returns:
            Value or None if not found
        """
        keys = key.split('.')
        current = self.data

        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
            else:
                return None

        return current

    def get_all(self) -> Dict:
        """Get entire structured memory."""
        return self.data.copy()

    def save(self) -> None:
        """Save to disk."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save structured memory: {e}")


# ================================================
# TESTS FOR EMBEDDING-DIM CHANGE ROBUSTNESS
# ================================================

def test_semantic_memory_embedding_dim_change():
    """
    Test that semantic memory correctly handles embedding dimension changes.
    Simulates switching embedding models (e.g., all-MiniLM to larger model).
    """
    import numpy as np

    # Create a test manager
    mem_mgr = MemoryManager()
    
    # Clear any existing memories for a clean test
    mem_mgr.semantic.clear()
    
    # Store initial embedding dim
    initial_dim = mem_mgr.semantic.embedding_dim
    logger.info(f"Initial embedding dim: {initial_dim}")

    # Add a memory entry
    mem_mgr.add_semantic_memory("Test memory 1", metadata={"source": "test"})
    assert len(mem_mgr.semantic.memories) == 1, f"Memory not added; found {len(mem_mgr.semantic.memories)} memories"
    logger.info(f"Added memory; stored embeddings have dim {len(mem_mgr.semantic.memories[0]['embedding'])}")

    # Search should work with current dim
    results = mem_mgr.semantic.search("Test memory")
    assert len(results) > 0, "Search failed with original dim"
    logger.info(f"Search found {len(results)} results with original dim")

    # Simulate a model change by forcing query_embedding to different dim
    # We'll mock the embedding to return a different dimension
    original_embed = mem_mgr.semantic.model_manager.embed
    
    def mock_embed_smaller(texts):
        """Mock embedder returning smaller dimension embeddings."""
        # Return embeddings of dim 128 (reduced from original)
        results = [np.random.randn(128).astype(np.float32) for _ in texts]
        return results

    # Patch embed to return smaller dim
    mem_mgr.semantic.model_manager.embed = mock_embed_smaller

    # Now search with the new dim
    # This should trigger re-embedding of stored memories to new dim
    try:
        results = mem_mgr.semantic.search("test query")
        logger.info(f"Search with different embedding dim succeeded; found {len(results)} results")
        # After re-embedding, the stored memories should have new dim
        assert mem_mgr.semantic.embedding_dim == 128, f"Embedding dim not updated: {mem_mgr.semantic.embedding_dim}"
        logger.info("✓ Embedding dimension auto-updated on search with different dim")
    except AssertionError as e:
        logger.error(f"✗ FAISS assertion error on search: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during search: {e}")
        raise
    finally:
        # Restore original embedder
        mem_mgr.semantic.model_manager.embed = original_embed

    # Test that add_memory also handles dim changes
    def mock_embed_larger(texts):
        """Mock embedder returning larger dimension embeddings."""
        # Return embeddings of dim 512 (much larger)
        results = [np.random.randn(512).astype(np.float32) for _ in texts]
        return results

    mem_mgr.semantic.model_manager.embed = mock_embed_larger
    
    try:
        mem_mgr.add_semantic_memory("Test memory 2", metadata={"source": "test"})
        logger.info(f"Added memory with new embedding dim; current stored dim: {mem_mgr.semantic.embedding_dim}")
        assert mem_mgr.semantic.embedding_dim == 512, f"Embedding dim not updated on add: {mem_mgr.semantic.embedding_dim}"
        logger.info("✓ Embedding dimension auto-updated on add_memory with different dim")
    except AssertionError as e:
        logger.error(f"✗ FAISS assertion error on add: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during add: {e}")
        raise
    finally:
        mem_mgr.semantic.model_manager.embed = original_embed

    logger.info("✓ All embedding-dim change tests passed")


if __name__ == "__main__":
    """Run robustness tests."""
    logger.info("Running semantic memory embedding-dim change tests...")
    try:
        test_semantic_memory_embedding_dim_change()
        logger.info("✓✓✓ All tests passed ✓✓✓")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
