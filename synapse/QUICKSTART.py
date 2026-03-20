"""
SYNAPSE - Quick Start Guide

Get up and running with SYNAPSE in 5 minutes.
"""

# =============================================================================
# STEP 1: Install Dependencies
# =============================================================================

# Run this command:
# pip install -r synapse/requirements.txt

# This installs:
# - openai>=1.3.0         (OpenAI SDK for OpenRouter API)
# - python-dotenv>=1.0.0  (Environment variable management)


# =============================================================================
# STEP 2: Get Your API Key
# =============================================================================

# 1. Go to: https://openrouter.ai/keys
# 2. Sign up / Log in
# 3. Navigate to API section
# 4. Create new API key
# 5. Copy your key


# =============================================================================
# STEP 3: Set Environment Variable
# =============================================================================

# Option A: Export in terminal (one-time)
# export DEEPSEEK_API_KEY="your-api-key-here"

# Option B: Create .env file in project root
# DEEPSEEK_API_KEY=your-api-key-here
# Then load with:
# from dotenv import load_dotenv
# load_dotenv()


# =============================================================================
# STEP 4: Run SYNAPSE CLI
# =============================================================================

# python -m synapse.main

# You'll see:
# ╔════════════════════════════════════════════════════════════╗
# ║                     SYNAPSE v1.0                           ║
# ║          JARVIS-Style AI Assistant with DeepSeek           ║
# ╚════════════════════════════════════════════════════════════╝
#
# 🚀 Initializing SYNAPSE...
# 🔗 Testing API connection...
# ✓ Connected successfully!
#
# Available Commands:
#   /help          - Show this help message
#   /memory        - Show conversation memory summary
#   /clear         - Clear conversation history
#   /export        - Export conversation memory
#   /status        - Test LLM connection
#   /exit          - Exit SYNAPSE


# =============================================================================
# STEP 5: Try Example Queries
# =============================================================================

# 👤 You: Hello, what can you do?
#
# ⏳ Processing...
#
# 🤖 Response:
# I'm SYNAPSE, your intelligent AI assistant. I can help you with...


# =============================================================================
# Quick Examples
# =============================================================================

print("""
EXAMPLE QUERIES TO TRY:

1. "Open Spotify"
   - Uses open_app tool

2. "Search web for 'machine learning tutorials'"
   - Uses search_web tool

3. "Read the requirements.txt file"
   - Uses read_file tool

4. "Create a file called test.txt with content: Hello world"
   - Uses write_file tool

5. "What's the capital of France?"
   - Regular query, no tools

6. "Show me my conversation memory"
   - Uses /memory command
""")


# =============================================================================
# Programmatic Quick Start
# =============================================================================

from synapse import SynapseCore
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    print("❌ DEEPSEEK_API_KEY not set!")
    exit(1)

# Initialize SYNAPSE
print("🚀 Starting SYNAPSE...")
synapse = SynapseCore(
    api_key=api_key,
    persist_memory=True,  # Save conversation
    debug=False           # Set to True for detailed logs
)

# Test connection
if not synapse.test_connection():
    print("❌ Failed to connect to DeepSeek API")
    exit(1)

print("✓ Connected!\n")

# Example 1: Simple query
print("=" * 60)
print("Example 1: Simple Query")
print("=" * 60)
result = synapse.process_query("Hello, what's your name?")
print(result["response"])
print()

# Example 2: Tool execution
print("=" * 60)
print("Example 2: Tool Execution")
print("=" * 60)
result = synapse.process_query("Search web for 'openrouter ai'")
if result["action"]:
    print(f"🔧 Action: {result['action']}")
    print(f"📊 Result: {result['action_result']}")
print()

# Example 3: Multi-turn conversation
print("=" * 60)
print("Example 3: Multi-turn Conversation")
print("=" * 60)
r1 = synapse.process_query("My name is Alice")
print(f"Q1: 'My name is Alice'\nA1: {r1['response']}\n")

r2 = synapse.process_query("What did I just tell you?")
print(f"Q2: 'What did I just tell you?'\nA2: {r2['response']}\n")

# Example 4: View memory
print("=" * 60)
print("Example 4: Memory Summary")
print("=" * 60)
summary = synapse.get_memory_summary()
print(f"Total messages: {summary['total_messages']}")
print(f"User queries: {summary['user_messages']}")
print(f"Assistant responses: {summary['assistant_messages']}")
print()

print("✅ Quick start complete!")


# =============================================================================
# Next Steps
# =============================================================================

print("""
NEXT STEPS:

1. Read synapse/README.md for full documentation
2. Check synapse/EXAMPLES.py for more code examples
3. Review synapse/IMPLEMENTATION.md for architecture details
4. Customize with your own system prompt
5. Add custom tools using ToolRegistry.register()
6. Deploy to production with proper error handling

KEY FILES:
- synapse/core.py           - Main orchestrator
- synapse/llm/deepseek_client.py - LLM integration
- synapse/tools/tool_registry.py - Tool system
- synapse/memory/memory.py  - Conversation history
- synapse/main.py           - CLI interface
""")
