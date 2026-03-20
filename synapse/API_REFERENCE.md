# SYNAPSE API Reference

## Overview

SYNAPSE is a modular AI assistant with DeepSeek integration. This document provides complete API reference for all public classes and methods.

---

## SynapseCore

**Module:** `synapse.core`

Main orchestrator class that coordinates LLM, tools, and memory.

### Constructor

```python
SynapseCore(
    api_key: str,
    system_prompt: Optional[str] = None,
    memory_max_messages: int = 20,
    persist_memory: bool = False,
    debug: bool = False,
)
```

**Parameters:**
- `api_key` (str): DeepSeek API key (required)
- `system_prompt` (str, optional): Custom system prompt. If None, uses default with tool descriptions
- `memory_max_messages` (int): Maximum messages to keep in memory (default: 20)
- `persist_memory` (bool): Save/load memory from disk (default: False)
- `debug` (bool): Enable debug logging (default: False)

**Example:**
```python
synapse = SynapseCore(
    api_key="sk-...",
    persist_memory=True,
    debug=False
)
```

### Methods

#### process_query

```python
def process_query(user_input: str) -> Dict[str, Any]
```

Process user query end-to-end and return response.

**Parameters:**
- `user_input` (str): User's input message

**Returns:** Dict with keys:
- `response` (str): Final response to user
- `thinking` (str|None): LLM reasoning/thinking
- `action` (str|None): Tool action name if executed
- `action_result` (str|None): Result from tool execution
- `complete` (bool): Whether query was fully processed

**Example:**
```python
result = synapse.process_query("Open Spotify")
print(result["response"])      # "I've opened Spotify..."
print(result["action"])         # "open_app"
print(result["action_result"])  # "Opened application: Spotify"
```

#### get_memory_summary

```python
def get_memory_summary() -> Dict[str, Any]
```

Get conversation memory summary.

**Returns:** Dict with:
- `total_messages` (int): Total messages in memory
- `user_messages` (int): User query count
- `assistant_messages` (int): Assistant response count
- `tools_used` (list): Tool actions executed
- `recent` (list): Last 3 interactions with role, preview, timestamp

**Example:**
```python
summary = synapse.get_memory_summary()
print(f"Total: {summary['total_messages']}")
print(f"Tools: {summary['tools_used']}")
```

#### clear_memory

```python
def clear_memory() -> None
```

Clear all conversation history.

**Example:**
```python
synapse.clear_memory()
```

#### test_connection

```python
def test_connection() -> bool
```

Test DeepSeek API connection and authentication.

**Returns:** True if connection successful, False otherwise

**Example:**
```python
if synapse.test_connection():
    print("Connected!")
else:
    print("Connection failed")
```

### Attributes

- `llm` (DeepSeekClient): LLM interface
- `tools` (ToolRegistry): Tool registry
- `memory` (ConversationMemory): Conversation memory
- `system_prompt` (str): Current system prompt

---

## DeepSeekClient

**Module:** `synapse.llm.deepseek_client`

OpenAI-compatible wrapper for DeepSeek API.

### Constructor

```python
DeepSeekClient(
    api_key: str,
    model: str = "deepseek-reasoner",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 120,
    max_tokens: int = 8000,
)
```

**Parameters:**
- `api_key` (str): DeepSeek API key
- `model` (str): Model name (default: "deepseek-reasoner")
- `base_url` (str): API base URL (default: "https://api.deepseek.com")
- `timeout` (int): Request timeout in seconds (default: 120)
- `max_tokens` (int): Max tokens in response (default: 8000)

### Methods

#### chat

```python
def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> LLMResponse
```

Send message to DeepSeek and get response.

**Parameters:**
- `messages` (list): Message dicts with 'role' and 'content'
- `temperature` (float): Temperature for generation 0-1 (default: 0.7)
- `system_prompt` (str, optional): System prompt to prepend

**Returns:** LLMResponse with:
- `content` (str): Response text
- `thinking` (str|None): LLM thinking (if available)
- `raw_message` (dict|None): Raw message object

**Raises:**
- `APIConnectionError`: If API connection fails
- `RateLimitError`: If rate limit exceeded
- `APIError`: General API errors

**Example:**
```python
client = DeepSeekClient(api_key="key")
response = client.chat(
    [{"role": "user", "content": "Hello"}],
    system_prompt="You are helpful"
)
print(response.content)
```

#### get_action_json

```python
def get_action_json(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]
```

Get structured JSON action from LLM response.

**Returns:** Parsed JSON dict with action and parameters

**Raises:** ValueError if response doesn't contain valid JSON

**Example:**
```python
json_response = client.get_action_json(
    [{"role": "user", "content": "Open Spotify"}]
)
print(json_response["action"])      # "open_app"
print(json_response["parameters"])  # {"app": "spotify"}
```

#### test_connection

```python
def test_connection() -> bool
```

Test API connection and authentication.

**Returns:** True if successful, False otherwise

**Example:**
```python
if client.test_connection():
    print("API ready")
```

---

## ToolRegistry

**Module:** `synapse.tools.tool_registry`

Registry of available tools and manual function-calling system.

### Constructor

```python
ToolRegistry()
```

Initializes with default tools (open_app, search_web, open_file, read_file, write_file).

### Methods

#### register

```python
def register(
    name: str,
    handler: Callable,
    description: str,
    parameters: Optional[Dict[str, str]] = None,
)
```

Register a new tool.

**Parameters:**
- `name` (str): Tool name (lowercase, no spaces)
- `handler` (callable): Function that executes the tool
- `description` (str): Human-readable description
- `parameters` (dict, optional): Parameter name -> description

**Example:**
```python
def translate(text, language):
    # Implementation
    return f"Translated to {language}"

registry.register(
    "translate",
    translate,
    "Translate text to another language",
    {
        "text": "Text to translate",
        "language": "Target language"
    }
)
```

#### execute

```python
def execute(action: str, parameters: Dict[str, Any]) -> str
```

Execute a tool by name with parameters.

**Parameters:**
- `action` (str): Tool name to execute
- `parameters` (dict): Tool parameters

**Returns:** String result from tool

**Raises:**
- `ValueError`: If tool not found
- `Exception`: If execution fails

**Example:**
```python
result = registry.execute("open_app", {"app": "spotify"})
print(result)  # "Opened application: Spotify"
```

#### get_tools_description

```python
def get_tools_description() -> str
```

Get formatted description of all available tools (for system prompt).

**Returns:** Formatted string

**Example:**
```python
desc = registry.get_tools_description()
print(desc)
```

#### get_tools_json_schema

```python
def get_tools_json_schema() -> List[Dict[str, Any]]
```

Get JSON schema of all tools (for LLM guidance).

**Returns:** List of tool schemas

**Example:**
```python
schemas = registry.get_tools_json_schema()
for schema in schemas:
    print(schema["action"])  # Tool name
```

### Attributes

- `tools` (dict): Tool name -> Tool objects

### Built-in Tools

#### open_app

```python
open_app(app: str) -> str
```

Open an application or URL.

**Parameters:**
- `app`: Application name or URL (e.g., "spotify", "https://example.com")

**Example:**
```python
result = registry.execute("open_app", {"app": "spotify"})
result = registry.execute("open_app", {"app": "https://google.com"})
```

#### search_web

```python
search_web(query: str) -> str
```

Search the web (opens in browser).

**Parameters:**
- `query`: Search query

**Example:**
```python
result = registry.execute("search_web", {"query": "machine learning"})
```

#### open_file

```python
open_file(path: str) -> str
```

Open a file with default application.

**Parameters:**
- `path`: File path (absolute or relative)

**Example:**
```python
result = registry.execute("open_file", {"path": "/Users/name/document.pdf"})
```

#### read_file

```python
read_file(path: str) -> str
```

Read file contents.

**Parameters:**
- `path`: File path

**Example:**
```python
result = registry.execute("read_file", {"path": "config.txt"})
print(result)  # File contents
```

#### write_file

```python
write_file(path: str, content: str) -> str
```

Write content to file.

**Parameters:**
- `path`: File path (creates if not exists)
- `content`: Content to write

**Example:**
```python
result = registry.execute("write_file", {
    "path": "output.txt",
    "content": "Hello world"
})
```

---

## ConversationMemory

**Module:** `synapse.memory.memory`

Lightweight conversation history management.

### Constructor

```python
ConversationMemory(
    max_messages: int = 20,
    persist_path: Optional[str] = None,
)
```

**Parameters:**
- `max_messages` (int): Max messages to keep (default: 20)
- `persist_path` (str, optional): Path to persist memory to disk

**Example:**
```python
memory = ConversationMemory(
    max_messages=50,
    persist_path="memory.json"
)
```

### Methods

#### add_message

```python
def add_message(
    role: str,
    content: str,
    action: Optional[str] = None,
    thinking: Optional[str] = None,
)
```

Add message to memory.

**Parameters:**
- `role` (str): "user", "assistant", or "system"
- `content` (str): Message content
- `action` (str, optional): Tool action if executed
- `thinking` (str, optional): LLM thinking/reasoning

**Example:**
```python
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi there!", action="no_action")
```

#### get_messages

```python
def get_messages(include_thinking: bool = False) -> List[Dict[str, str]]
```

Get messages in OpenAI-compatible format.

**Parameters:**
- `include_thinking` (bool): Include thinking field if available

**Returns:** List of message dicts with 'role' and 'content'

**Example:**
```python
messages = memory.get_messages()
# Returns: [
#   {"role": "user", "content": "Hello"},
#   {"role": "assistant", "content": "Hi there!"}
# ]
```

#### get_summary

```python
def get_summary() -> Dict[str, Any]
```

Get memory summary statistics.

**Returns:** Dict with counts and recent interactions

**Example:**
```python
summary = memory.get_summary()
print(summary)
# {
#   "total_messages": 2,
#   "user_messages": 1,
#   "assistant_messages": 1,
#   "tools_used": [],
#   "recent": [...]
# }
```

#### clear

```python
def clear() -> None
```

Clear all messages.

**Example:**
```python
memory.clear()
```

#### save

```python
def save() -> None
```

Save memory to disk (if persist_path is set).

**Example:**
```python
memory.save()
```

#### load

```python
def load() -> None
```

Load memory from disk (if persist_path is set).

**Example:**
```python
memory.load()
```

### Attributes

- `messages` (list): Message objects
- `max_messages` (int): Maximum messages to keep
- `persist_path` (str|None): Disk storage path

---

## Message

**Module:** `synapse.memory.memory`

Represents a single message.

### Attributes

- `role` (str): "user", "assistant", or "system"
- `content` (str): Message text
- `timestamp` (str): ISO format timestamp
- `action` (str|None): Tool action name if any
- `thinking` (str|None): LLM reasoning/thinking

---

## LLMResponse

**Module:** `synapse.llm.deepseek_client`

Structured response from DeepSeek LLM.

### Attributes

- `content` (str): Response text
- `thinking` (str|None): LLM thinking (if available)
- `raw_message` (dict|None): Raw message object

---

## Tool

**Module:** `synapse.tools.tool_registry`

Represents a callable tool.

### Attributes

- `name` (str): Tool name
- `description` (str): Tool description
- `handler` (callable): Function that executes the tool
- `parameters` (dict): Parameter name -> description

---

## Exceptions

### APIConnectionError

Connection to DeepSeek API failed.

```python
from openai import APIConnectionError
```

### APIError

General API error from DeepSeek.

```python
from openai import APIError
```

### RateLimitError

Rate limit exceeded on DeepSeek API.

```python
from openai import RateLimitError
```

---

## Constants

```python
# Default model
DEFAULT_MODEL = "deepseek-reasoner"

# Default API base URL
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Default system prompt includes tool descriptions
DEFAULT_SYSTEM_PROMPT = "You are SYNAPSE..."

# Default memory size
DEFAULT_MAX_MESSAGES = 20

# Default token limit
DEFAULT_MAX_TOKENS = 8000

# Default timeout
DEFAULT_TIMEOUT = 120
```

---

## Environment Variables

- `DEEPSEEK_API_KEY`: DeepSeek API key (required)
- `SYNAPSE_DEBUG`: Enable debug logging (true/false)
- `SYNAPSE_MAX_MESSAGES`: Memory size (integer)

---

## Type Hints

```python
from typing import Dict, Any, List, Optional, Callable

# Tool handler type
ToolHandler = Callable[..., str]

# Message type (for get_messages)
Message = Dict[str, str]  # {"role": str, "content": str}

# Action response type
Action = Dict[str, Any]  # {"action": str, "parameters": Dict}

# Query response type
QueryResult = Dict[str, Any]
# {
#   "response": str,
#   "thinking": Optional[str],
#   "action": Optional[str],
#   "action_result": Optional[str],
#   "complete": bool
# }
```

---

## Common Patterns

### Initialize and Test

```python
from synapse import SynapseCore
import os
from dotenv import load_dotenv

load_dotenv()
synapse = SynapseCore(api_key=os.getenv("DEEPSEEK_API_KEY"))

if synapse.test_connection():
    print("Ready!")
```

### Add Custom Tool

```python
def my_tool(param1, param2):
    return f"Result: {param1} {param2}"

synapse.tools.register(
    "my_tool",
    my_tool,
    "Description",
    {"param1": "desc1", "param2": "desc2"}
)
```

### Process Query with Error Handling

```python
try:
    result = synapse.process_query("Hello")
    if result["complete"]:
        print(result["response"])
    else:
        print(f"Error: {result['response']}")
except Exception as e:
    print(f"Failed: {e}")
```

### View Memory

```python
summary = synapse.get_memory_summary()
print(f"Messages: {summary['total_messages']}")
print(f"Tools used: {summary['tools_used']}")
```

### Export Conversation

```python
import json

messages = synapse.memory.get_messages(include_thinking=True)
with open("conversation.json", "w") as f:
    json.dump(messages, f, indent=2)
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `No API key` | Missing DEEPSEEK_API_KEY | Set environment variable or pass api_key |
| `Connection failed` | Invalid API key or no internet | Verify key, check network, try `/status` |
| `Invalid JSON` | LLM response not JSON | Normal for non-action responses, treated as text |
| `Tool not found` | Unknown tool accessed | Register tool first with `register()` |
| `Memory full` | Reached max_messages | Increase max_messages or clear memory |

