"""
═════════════════════════════════════════════════════════════════════════════
4-LAYER MEMORY ARCHITECTURE - IMPLEMENTATION GUIDE
═════════════════════════════════════════════════════════════════════════════

OVERVIEW
────────────────────────────────────────────────────────────────────────────

The LocalAI system now implements a sophisticated 4-layer memory architecture
that manages information at different granularities and timeframes.


ARCHITECTURE LAYERS
════════════════════════════════════════════════════════════════════════════

1. SHORT-TERM MEMORY (Layer 1)
──────────────────────────────────────────────────────────────────────────
Purpose: Active conversation context
Storage: In-memory deque with configurable max size
Lifecycle: Cleared on new conversation or when limit reached
Config: SHORT_TERM_CONFIG in config.yaml

Features:
  • Stores last N messages (default: 10)
  • Immediate access for prompt construction
  • Automatically discards oldest messages when full
  • Contains role + content + timestamp

Usage:
  memory_manager.add_short_term_message('user', "Hello")
  context = memory_manager.get_short_term_context()


2. ROLLING SUMMARY MEMORY (Layer 2)
──────────────────────────────────────────────────────────────────────────
Purpose: Compressed conversation history
Storage: JSON list of summaries
Lifecycle: Created periodically, old summaries discarded
Config: ROLLING_SUMMARY_CONFIG in config.yaml

Features:
  • Automatically summarizes when messages exceed threshold (default: 15)
  • Extracts key facts, decisions, goals, unresolved topics
  • Keeps recent summaries for context (default: 3)
  • Preserves important conversational context

Summarization includes:
  • Facts: Messages with important information
  • Decisions: Messages indicating conclusions/preferences
  • Goals: Messages about user goals and aspirations
  • Unresolved: Questions or unclear topics needing attention
  • Key phrases: Important concepts and terms

Usage:
  created = memory_manager.maybe_create_summary()
  context = memory_manager.get_rolling_summary_context()


3. SEMANTIC MEMORY (Layer 3)
──────────────────────────────────────────────────────────────────────────
Purpose: Vector-indexed persistent facts and insights
Storage: JSON embeddings + FAISS index
Lifecycle: Persistent across sessions
Config: SEMANTIC_CONFIG in config.yaml

Features:
  • Uses all-MiniLM-L6-v2 embeddings (384-dimensional)
  • FAISS index for fast similarity search
  • Metadata support (source, type, timestamp)
  • Similarity-based retrieval with configurable threshold
  • Dynamic index updates

Files:
  • semantic_memory.json - embedding vectors and metadata
  • semantic_index.faiss - FAISS index for fast search

Usage:
  memory_manager.add_semantic_memory("Important fact about user")
  results = memory_manager.search_semantic_memory("user's interests")
  context = memory_manager.get_semantic_context(user_query)


4. STRUCTURED PERSISTENT MEMORY (Layer 4)
──────────────────────────────────────────────────────────────────────────
Purpose: User profile and system state
Storage: JSON file with dot-notation key access
Lifecycle: Persistent across sessions
Config: STRUCTURED_CONFIG in config.yaml

Categories:
  • user: Name, age, birth year, contact info
  • preferences: Communication style, interests, preferences
  • goals: Long-term goals and aspirations
  • system_state: Important system variables

Features:
  • Auto-extraction from user messages
  • Dot-notation access: "user.name", "preferences.learning_style"
  • Structured JSON storage for easy integration
  • Automatic updates from conversation analysis

Usage:
  memory_manager.update_structured_memory("user.name", "Alice")
  name = memory_manager.get_structured_memory("user.name")
  profile = memory_manager.get_structured_memory()  # Get all
  memory_manager.extract_profile_facts("My name is Bob and I'm 25")


CONFIGURATION
════════════════════════════════════════════════════════════════════════════

See config.yaml for all tunable parameters:

memory:
  short_term:
    max_messages: 10              # Active conversation window
    
  rolling_summary:
    enabled: true
    trigger_threshold: 15         # Summarize at this many messages
    summary_size: 3               # Keep N recent summaries
    preserve_facts: true
    
  semantic:
    max_results: 3                # Results per search
    similarity_threshold: 0.5      # Min similarity (0-1)
    
  structured:
    enabled: true
    file: "structured_memory.json"

summarization:
  model_type: "extractive"        # Lightweight, no external model needed
  key_phrases_count: 5
  preserve_unresolved: true

search:
  default_top_k: 3
  include_summaries: true
  include_semantic: true


INTEGRATION WITH GENERATION PIPELINE
════════════════════════════════════════════════════════════════════════════

The AIService class now uses the 4-layer memory for all generation:

1. User input arrives
2. Added to short-term memory
3. Profile facts auto-extracted
4. Check if rolling summary needed
5. Build multi-layer context:
   - Short-term: Active conversation
   - Rolling summary: Historical context
   - Semantic: Relevant facts via similarity search
   - Structured: User profile and preferences
6. Construct comprehensive prompt with all context
7. Generate response
8. Add response to short-term memory
9. Save all persistent memory

Flow diagram:
  User Input
    ↓
  ✓ Add to short-term
  ✓ Extract profile facts
  ✓ Maybe create rolling summary
  ✓ Search semantic memory
  ✓ Fetch user profile
    ↓
  Build full context (4 layers)
  Construct prompt with ChatFormatter
    ↓
  Generate response
    ↓
  ✓ Add assistant response to short-term
  ✓ Save all layers
    ↓
  Return response


FILE LOCATIONS
════════════════════════════════════════════════════════════════════════════

Core files:
  core/chat_formatter.py        - Qwen2.5 chat template formatting
  core/model_manager.py         - Model loading and generation
  core/config.py                - Config loading and constants

Memory files:
  memory/memory_manager.py      - 4-layer coordinator (MAIN)
  memory/summarizer.py          - Rolling summary generation
  memory/semantic_memory.py     - Vector embedding index
  memory/memory_store.py        - Legacy (for compatibility)

Persistent storage:
  config.yaml                   - Configuration (EDIT THIS)
  structured_memory.json        - User profile and state
  semantic_memory.json          - Fact embeddings
  semantic_index.faiss          - FAISS vector index


BACKWARD COMPATIBILITY
════════════════════════════════════════════════════════════════════════════

The AIService supports both systems:

Default:    use_4layer_memory=True   (NEW 4-layer system)
Legacy:     use_4layer_memory=False  (Old memory_store)

To use legacy system:
  ai_service = AIService(use_4layer_memory=False)

The code automatically falls back if needed, so existing integrations
continue to work without modification.


MEMORY CLEANUP AND MANAGEMENT
════════════════════════════════════════════════════════════════════════════

Dynamic memory management:
  • Short-term: Auto-discards oldest when reaching max_messages
  • Rolling summaries: Keeps only recent summaries
  • Semantic: Persistent, manual cleanup via delete_memory()
  • Structured: Persistent, manual updates via update_structured_memory()

To clear for new user:
  memory_manager.clear_short_term()

To save all memory:
  memory_manager.save_all()

Memory size monitoring:
  info = memory_manager.semantic.get_info()
  # Returns: total_memories, embedding_dimension, index_exists


PERFORMANCE CONSIDERATIONS
════════════════════════════════════════════════════════════════════════════

Short-term: O(1) append, O(N) context construction
Rolling summary: O(N) extraction, O(1) access
Semantic: O(log N) search (FAISS), O(N) add/delete
Structured: O(1) get/set (dict access)

Optimization tips:
  • Reduce rolling_summary trigger_threshold for more frequent summaries
  • Reduce short_term max_messages for smaller context windows
  • Reduce semantic max_results to process fewer similarity matches
  • Set similarity_threshold higher to filter weak matches


EXTENDING THE SYSTEM
════════════════════════════════════════════════════════════════════════════

Adding new memory layers:
1. Create new module in memory/ folder
2. Implement same interface as SemanticMemory
3. Add to MemoryManager.__init__
4. Add configuration section to config.yaml
5. Integrate in build_full_context()

Customizing summarization:
1. Modify ConversationSummarizer.KEY_INDICATORS
2. Adjust extraction logic in _extract_key_information
3. Customize summary_text formatting

Upgrading embedding model:
1. Update SEMANTIC_CONFIG in config.yaml
2. Modify SemanticMemory.embedding_dim
3. Rebuild all semantic indices


TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════

Issue: Summary not being created
  → Check config.yaml rolling_summary.trigger_threshold
  → Verify you have enough messages queued
  
Issue: Semantic search returns empty
  → Check if semantic.enabled = true in config.yaml
  → Verify semantic_memory.json exists with entries
  → Check similarity_threshold isn't too high

Issue: Memory files growing too large
  → Reduce summary_size to keep fewer summaries
  → Reduce short_term max_messages
  → Implement archival logic for old semantic memories

Issue: Slow generation
  → Reduce semantic max_results
  → Reduce short_term max_messages
  → Profile with semantic.get_info()


════════════════════════════════════════════════════════════════════════════
"""