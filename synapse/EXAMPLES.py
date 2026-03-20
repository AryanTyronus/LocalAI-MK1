"""Example usage patterns for SYNAPSE."""

# Example 1: Basic CLI Usage
# ==========================
# Run this in your terminal:
#
# python -m synapse.main
#
# Then type:
# 👤 You: Open Spotify
# 👤 You: Search web for "openrouter api documentation"
# 👤 You: Read the requirements.txt file
# 👤 You: /memory
# 👤 You: /exit


# Example 2: Programmatic Usage
# ==============================

from synapse import SynapseCore

# Initialize
synapse = SynapseCore(
    api_key="your-deepseek-api-key",
    persist_memory=True,
    debug=False
)

# Test connection
if not synapse.test_connection():
    print("Failed to connect!")
    exit()

# Process a query
result = synapse.process_query("Open Spotify and play jazz music")

print(f"Response: {result['response']}")
print(f"Thinking: {result['thinking']}")
print(f"Action: {result['action']}")
print(f"Action Result: {result['action_result']}")

# View memory
memory_summary = synapse.get_memory_summary()
print(f"Conversation summary: {memory_summary}")


# Example 3: Register Custom Tool
# ================================

from synapse import SynapseCore

synapse = SynapseCore(api_key="your-key")

def calculate_discount(price, discount_percent):
    """Calculate discount on a price."""
    discounted = price * (1 - discount_percent / 100)
    return f"Price: ${price}, Discount: {discount_percent}% → Final: ${discounted:.2f}"

# Register the tool
synapse.tools.register(
    "calculate_discount",
    calculate_discount,
    "Calculate discounted price",
    {
        "price": "Original price (number)",
        "discount_percent": "Discount percentage (0-100)"
    }
)

# Now SYNAPSE can use it
result = synapse.process_query("What's 20% off of $100?")
print(result["response"])


# Example 4: Custom System Prompt
# ================================

custom_prompt = """
You are SYNAPSE, an AI assistant specialized in software development.

Your capabilities:
- Help with code reviews
- Debug Python/JavaScript code
- Suggest architecture improvements
- Answer technical questions

When you need to take action, use the available tools.
Always explain your reasoning.
"""

synapse = SynapseCore(
    api_key="your-key",
    system_prompt=custom_prompt,
    memory_max_messages=30
)

result = synapse.process_query(
    "I have a Python function that's slow. Can you help?"
)
print(result["response"])


# Example 5: Multi-turn Conversation
# ===================================

synapse = SynapseCore(api_key="your-key", persist_memory=True)

# First interaction
q1 = synapse.process_query("What's my name?")
print(f"Q1: {q1['response']}\n")

# Second interaction - should remember context
q2 = synapse.process_query("I'm Alice. Nice to meet you.")
print(f"Q2: {q2['response']}\n")

# Third interaction - should reference earlier messages
q3 = synapse.process_query("What did I tell you earlier?")
print(f"Q3: {q3['response']}\n")

# View conversation
summary = synapse.get_memory_summary()
print(f"Total messages: {summary['total_messages']}")
print(f"Recent interactions:")
for msg in summary['recent']:
    print(f"  - {msg['role']}: {msg['preview']}")


# Example 6: Error Handling
# ========================

from synapse import SynapseCore
import sys

try:
    synapse = SynapseCore(
        api_key="invalid-key",
        debug=True  # Enable debug to see what's happening
    )
    
    # This will fail gracefully
    result = synapse.process_query("Hello")
    
    if result["complete"]:
        print(result["response"])
    else:
        print(f"Failed: {result['response']}")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)


# Example 7: Memory Management
# ============================

synapse = SynapseCore(api_key="your-key", persist_memory=True)

# Process multiple queries
for i in range(5):
    synapse.process_query(f"Query number {i+1}")

# Check memory
summary = synapse.get_memory_summary()
print(f"Messages in memory: {summary['total_messages']}")
print(f"User queries: {summary['user_messages']}")
print(f"Assistant responses: {summary['assistant_messages']}")

# Export memory
import json
messages = synapse.memory.get_messages(include_thinking=True)
with open("conversation_export.json", "w") as f:
    json.dump(messages, f, indent=2)

# Clear if needed
synapse.clear_memory()


# Example 8: Batch Processing
# ============================

synapse = SynapseCore(api_key="your-key")

queries = [
    "Help me debug this code",
    "What are best practices?",
    "Show me an example",
]

for query in queries:
    result = synapse.process_query(query)
    print(f"Q: {query}")
    print(f"A: {result['response']}\n")
    print("---\n")


# Example 9: Environment Configuration
# ===============================

import os

# Set via environment variables
os.environ["DEEPSEEK_API_KEY"] = "your-key"

# Or use .env file with python-dotenv
from dotenv import load_dotenv
load_dotenv()

synapse = SynapseCore(api_key=os.getenv("DEEPSEEK_API_KEY"))


# Example 10: Production Setup
# =============================

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synapse.log'),
        logging.StreamHandler()
    ]
)

# Initialize with production settings
synapse = SynapseCore(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    system_prompt=None,  # Use default
    memory_max_messages=50,
    persist_memory=True,  # Save to disk
    debug=False  # Production level logging
)

# Ready for use
result = synapse.process_query("Hello, SYNAPSE")
logging.info(f"Response: {result['response']}")
