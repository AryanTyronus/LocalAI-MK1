"""
4-LAYER MEMORY SYSTEM - USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════

This file shows practical examples of using the new 4-layer memory system.
"""

# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: BASIC USAGE WITH AIService
# ═════════════════════════════════════════════════════════════════════════

from services.ai_service import AIService

# Initialize AIService with new 4-layer memory (default)
ai_service = AIService(use_4layer_memory=True)

# Simple conversation
response1 = ai_service.ask("Hello! My name is Alice and I'm 25 years old.")
print(f"Response: {response1}")

response2 = ai_service.ask("What are my interests?")
print(f"Response: {response2}")

# The system automatically:
# ✓ Added your message to short-term memory
# ✓ Extracted "Alice" and "25" to structured memory
# ✓ Created rolling summary if threshold reached
# ✓ Searched for relevant facts in semantic memory
# ✓ Built comprehensive context from all layers


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: DIRECT MEMORY MANAGER USAGE
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

# Create memory manager
memory = MemoryManager()

# Layer 1: Short-term memory
memory.add_short_term_message('user', "I like physics and math")
memory.add_short_term_message('assistant', "Great! I can help with physics and math.")
context = memory.get_short_term_context()
print("Short-term context:\n", context)

# Layer 2: Rolling summary (automatic or manual)
was_created = memory.maybe_create_summary()
if was_created:
    print("Rolling summary created!")
summary_context = memory.get_rolling_summary_context()
print("Summary context:\n", summary_context)

# Layer 3: Semantic memory
memory.add_semantic_memory("Alice loves physics and enjoys problem solving")
memory.add_semantic_memory("Alice's goals: Master advanced calculus")
results = memory.search_semantic_memory("What does Alice like?")
print("Semantic search results:", results)

# Layer 4: Structured memory
memory.update_structured_memory("user.name", "Alice")
memory.update_structured_memory("user.age", 25)
memory.update_structured_memory("preferences.subject", "Physics")
memory.update_structured_memory("goals", ["Master calculus", "Learn quantum mechanics"])

name = memory.get_structured_memory("user.name")
profile = memory.get_structured_memory()  # Get everything
print(f"User name from structured memory: {name}")
print(f"Full profile: {profile}")

# Save everything to disk
memory.save_all()


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: BUILDING FULL CONTEXT FOR GENERATION
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

memory = MemoryManager()

# Add some conversation
memory.add_short_term_message('user', "Tell me about black holes")
memory.add_short_term_message('assistant', "Black holes are regions of spacetime...")

# Build complete context from all layers
full_context = memory.build_full_context("What's at the center of a black hole?")

print("FULL CONTEXT LAYERS:")
print("─" * 50)
print("Short-term:\n", full_context['short_term'][:100] + "...")
print("\nRolling summary:\n", full_context['rolling_summary'][:100] + "...")
print("\nSemantic:\n", full_context['semantic'][:100] + "...")
print("\nStructured:\n", full_context['structured'][:100] + "...")


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: AUTO-EXTRACTING PROFILE FROM CONVERSATION
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

memory = MemoryManager()

# Extract profile from natural language
memory.extract_profile_facts("My name is Bob and I was born in 1998")
memory.extract_profile_facts("I'm 26 years old and I love programming")

# Retrieved automatic extraction
name = memory.get_structured_memory("user.name")
age = memory.get_structured_memory("user.age")
birth_year = memory.get_structured_memory("user.birth_year")

print(f"Name: {name}")
print(f"Age: {age}")
print(f"Birth year: {birth_year}")


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: SEMANTIC MEMORY SEARCH WITH SCORES
# ═════════════════════════════════════════════════════════════════════════

from memory.semantic_memory import SemanticMemory
from core.model_manager import ModelManager

model_manager = ModelManager.get_instance()
semantic = SemanticMemory(model_manager)

# Add facts
semantic.add_memory("Alice enjoys solving differential equations")
semantic.add_memory("Alice's favorite subject is physics")
semantic.add_memory("Alice studied at MIT")

# Search with similarity scores
results = semantic.search_with_scores("Tell me about Alice", k=2)
print("Search results with scores:")
for text, score in results:
    print(f"  [{score:.3f}] {text}")


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: ROLLING SUMMARY CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

# The summarizer automatically extracts:
# • Facts - "My name is", "I was born"
# • Decisions - "I prefer", "I decided"
# • Goals - "I want to", "My goal is"
# • Unresolved - "I'm confused", "How do I?"
# • Key phrases - Important concepts

from memory.summarizer import ConversationSummarizer

summarizer = ConversationSummarizer()

messages = [
    {"role": "user", "content": "My name is Charlie. I want to learn machine learning."},
    {"role": "assistant", "content": "Great! ML is fascinating."},
    {"role": "user", "content": "I'm confused about backpropagation though."},
    {"role": "assistant", "content": "Let me explain the chain rule..."},
]

summary = summarizer.summarize_messages(messages)
print("Summary:")
print(f"  Facts: {summary['facts']}")
print(f"  Goals: {summary['goals']}")
print(f"  Unresolved: {summary['unresolved_topics']}")
print(f"  Key phrases: {summary['key_phrases']}")
print(f"  Summary text: {summary['summary_text']}")


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: CONFIGURATION & SETUP
# ═════════════════════════════════════════════════════════════════════════

# Edit config.yaml to adjust behavior:

"""
memory:
  short_term:
    max_messages: 10          # ← Increase for longer context windows
    
  rolling_summary:
    trigger_threshold: 15     # ← Lower = more frequent summaries
    summary_size: 3           # ← Keep more/fewer old summaries
    
  semantic:
    max_results: 3            # ← Return more results
    similarity_threshold: 0.5  # ← Higher = stricter matching
"""

# After editing config.yaml, these changes take effect on next initialization:
from importlib import reload
import core.config
reload(core.config)

from memory.memory_manager import MemoryManager
memory = MemoryManager()  # Uses updated config


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: PERSISTENCE ACROSS SESSIONS
# ═════════════════════════════════════════════════════════════════════════

# All 4 layers are automatically persistent:

# Session 1:
memory1 = MemoryManager()
memory1.add_semantic_memory("User likes quantum physics")
memory1.update_structured_memory("user.name", "Diana")
memory1.save_all()

# Session 2 (next time app starts):
memory2 = MemoryManager()
# All semantic memories from session 1 are loaded!
# All structured memory from session 1 is loaded!
results = memory2.search_semantic_memory("user interests")
name = memory2.get_structured_memory("user.name")
print(f"Loaded name: {name}")  # "Diana"


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 9: CLEARING MEMORY FOR NEW USER
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

memory = MemoryManager()

# Clear short-term for new conversation topic
memory.clear_short_term()

# Note: To fully reset for new user, you'd need:
# - Reset short-term: memory.clear_short_term()
# - Keep summaries (historical context) or clear manually
# - Reset structured: Create fresh StructuredMemory
# - Clear semantic: semantic.clear()


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 10: MONITORING MEMORY STATE
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

memory = MemoryManager()

# Check semantic memory size
info = memory.semantic.get_info()
print(f"Semantic memories: {info['total_memories']}")
print(f"Embedding dimension: {info['embedding_dimension']}")
print(f"Index exists: {info['index_exists']}")

# Check what's in short-term
messages = memory.get_short_term_messages()
print(f"Short-term messages: {len(messages)}")

# Check summaries
print(f"Rolling summaries: {len(memory.rolling_summaries)}")

# Check profile
profile = memory.get_structured_memory()
print(f"User profile keys: {list(profile.keys())}")


# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE 11: CUSTOM METADATA FOR SEMANTIC MEMORY
# ═════════════════════════════════════════════════════════════════════════

from memory.memory_manager import MemoryManager

memory = MemoryManager()

# Add semantic memory with metadata
memory.add_semantic_memory(
    "Alice achieved 95% in physics exam",
    metadata={
        'source': 'achievement',
        'subject': 'physics',
        'importance': 'high',
        'date': '2026-02-24'
    }
)

# Later retrieve with metadata preserved
results = memory.semantic.search("Alice's achievements", k=1)
memory_entry = memory.semantic.get_memory(0)
print(f"Text: {memory_entry['text']}")
print(f"Metadata: {memory_entry['metadata']}")


# ═════════════════════════════════════════════════════════════════════════
# TIPS & BEST PRACTICES
# ═════════════════════════════════════════════════════════════════════════

"""
1. MEMORY LAYER SELECTION
   • Use short-term for immediate context
   • Use rolling summaries for historical patterns
   • Use semantic for facts and insights
   • Use structured for user profile

2. CONFIGURATION TUNING
   • Increase short_term.max_messages for longer context windows
   • Lower rolling_summary.trigger_threshold for better history compression
   • Adjust semantic.max_results based on query quality
   • Set similarity_threshold to balance precision vs recall

3. PERFORMANCE
   • Monitor memory size with semantic.get_info()
   • Archive old semantic memories periodically
   • Keep summaries recent (don't increase summary_size too much)
   • Use similarity_threshold > 0.5 for higher quality results

4. INTEGRATION WITH GENERATION
   • Always call memory_manager.save_all() after generation
   • Use extract_profile_facts() to maintain user model
   • Call maybe_create_summary() to manage history
   • Use build_full_context() for comprehensive prompt assembly

5. DEBUGGING
   • Enable logging to see memory operations
   • Check files: structured_memory.json, semantic_memory.json
   • Use get_info() to verify index state
   • Print context before generation for inspection
"""
