# SYNAPSE Implementation Guide

## Project Structure

```
synapse/
├── __init__.py              # Package export
├── core.py                  # SynapseCore orchestrator
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── README.md               # Usage documentation
├── EXAMPLES.py             # Code examples
├── IMPLEMENTATION.md       # This file
├── config.yaml            # Configuration template
│
├── llm/
│   ├── __init__.py
│   └── deepseek_client.py  # OpenAI-compatible wrapper
│
├── tools/
│   ├── __init__.py
│   └── tool_registry.py    # Tool system & registry
│
└── memory/
    ├── __init__.py
    └── memory.py           # Conversation memory
```

## Core Architecture

### 1. Request Flow

```
User Input
    ↓
SynapseCore.process_query()
    ↓
Memory.add_message(user)
    ↓
DeepSeekClient.chat()
    ↓
LLM Response (with optional JSON action)
    ↓
Try to parse JSON?
    ├─ YES: Extract action & parameters
    │   ↓
    │   ToolRegistry.execute()
    │   ↓
    │   Get result
    │   ↓
    │   LLM follow-up: "Based on result..."
    │   ↓
    │   Final response
    │
    └─ NO: Use response directly
    ↓
Memory.add_message(assistant)
    ↓
Return result to user
```

### 2. Manual Function Calling

Since DeepSeek doesn't support native function calling, we:
1. Instruct LLM to return JSON with `{"action": "...", "parameters": {...}}`
2. Parse the JSON response
3. Look up tool in registry
4. Execute with parameters
5. Send result back to LLM for contextual response

**Example LLM Response:**
```json
{
  "action": "open_app",
  "parameters": {"app": "spotify"},
  "reasoning": "User wants to open Spotify to play music"
}
```

### 3. Memory Management

- Stores up to N messages (default: 20)
- Maintains full conversation context
- Optional disk persistence
- Efficient for context windows
- Trims oldest messages when full

## Component Details

### DeepSeekClient

**Purpose:** Abstraction layer over DeepSeek API

**Key Methods:**
- `chat()` - Send message and get response
- `get_action_json()` - Get structured JSON action
- `test_connection()` - Verify API access
- `_parse_json()` - Parse JSON from response text

**Error Handling:**
- APIConnectionError → Connection failed
- RateLimitError → Rate limit exceeded
- APIError → General API error
- JSONDecodeError → Invalid JSON response

**Configuration:**
```python
client = DeepSeekClient(
    api_key="key",
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    timeout=120,
    max_tokens=8000
)
```

### SynapseCore

**Purpose:** Central orchestrator managing all components

**Key Methods:**
- `process_query()` - Main entry point
- `get_memory_summary()` - Memory statistics
- `clear_memory()` - Reset conversation
- `test_connection()` - Verify setup

**Initialization:**
```python
synapse = SynapseCore(
    api_key="key",
    system_prompt=None,  # Optional
    memory_max_messages=20,
    persist_memory=False,
    debug=False
)
```

### ToolRegistry

**Purpose:** Manage available tools and execute them

**Key Methods:**
- `register()` - Add new tool
- `execute()` - Run tool by name
- `get_tools_description()` - For system prompt
- `get_tools_json_schema()` - For JSON guidance

**Built-in Tools:**
- `open_app(app)` - Opens application or URL
- `search_web(query)` - Opens web search
- `open_file(path)` - Opens file with default app
- `read_file(path)` - Reads and returns file content
- `write_file(path, content)` - Writes to file

**Adding Custom Tool:**
```python
registry.register(
    name="my_tool",
    handler=my_function,
    description="What it does",
    parameters={"param": "description"}
)
```

### ConversationMemory

**Purpose:** Maintain conversation history and context

**Key Methods:**
- `add_message()` - Add message to memory
- `get_messages()` - Get in OpenAI format
- `get_summary()` - Memory statistics
- `clear()` - Clear all messages
- `save()` / `load()` - Persistence

**Message Format:**
```python
Message(
    role="user|assistant|system",
    content="Message text",
    timestamp="ISO timestamp",
    action="Tool name (if executed)",
    thinking="LLM reasoning (if available)"
)
```

## System Prompt Strategy

The system prompt guides the LLM to:
1. Understand available tools
2. Respond with JSON when action needed
3. Explain reasoning
4. Be helpful and concise

**Default Template:**
```
You are SYNAPSE, a powerful AI assistant.

You have access to the following tools:
[list of tools with descriptions]

When you need to execute an action, respond with a JSON object like this:
{
  "action": "tool_name",
  "parameters": {"param1": "value1", "param2": "value2"},
  "reasoning": "Why you're doing this"
}

Always provide clear explanations for your actions.
Be concise and helpful.
```

## Error Handling Strategy

```python
Try:
  Call LLM
Catch APIError:
  Return error to user
  Log error
  
Try:
  Parse JSON from response
Catch JSONDecodeError:
  Treat as regular text response
  
Try:
  Execute tool
Catch ValueError (unknown tool):
  Report unknown action
Catch Exception (execution failed):
  Report tool error
  Send error to LLM for context
```

## Performance Considerations

- **API Calls:** ~1 per query (+ 1 follow-up if tool executed)
- **Memory Usage:** ~50MB initialization, grows with messages
- **Token Usage:** System prompt + last N messages + current query
- **Latency:** 1-5 seconds (depends on DeepSeek API response time)
- **Storage:** Optional disk persistence (~1KB per message)

## Extension Points

### Add Custom Tool
```python
def translate(text, target_language):
    # Implementation
    return translated_text

synapse.tools.register("translate", translate, "...", {...})
```

### Custom System Prompt
```python
custom = """You are a Python debugging expert..."""
synapse = SynapseCore(api_key="key", system_prompt=custom)
```

### Extend Memory
```python
# Store additional metadata
class ExtendedMessage(Message):
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
```

### Add Pre/Post Processing
```python
# Before sending to LLM
processed = preprocess_message(user_input)
result = synapse.process_query(processed)

# After receiving result
formatted = format_output(result)
print(formatted)
```

## Testing Strategy

```python
# Unit test individual components
def test_deepseek_client():
    client = DeepSeekClient(api_key="test")
    assert client.test_connection()

def test_tool_registry():
    registry = ToolRegistry()
    result = registry.execute("search_web", {"query": "test"})
    assert result

def test_memory():
    memory = ConversationMemory()
    memory.add_message("user", "Hello")
    messages = memory.get_messages()
    assert len(messages) == 1

# Integration test full flow
def test_synapse_core():
    synapse = SynapseCore(api_key="test")
    result = synapse.process_query("Open Spotify")
    assert result["complete"]
    assert result["response"]
```

## Deployment Considerations

1. **Environment Variables:** Set `DEEPSEEK_API_KEY`
2. **Logging:** Configure via `logging.basicConfig()`
3. **Memory Persistence:** Enable with `persist_memory=True`
4. **Resource Limits:** Set `memory_max_messages` based on available RAM
5. **Error Handling:** Wrap `process_query()` in try-except
6. **Rate Limiting:** DeepSeek API has rate limits; implement retry logic if needed

## Future Enhancements

- [ ] Async/await support
- [ ] Tool chaining (multiple sequential tools)
- [ ] Function calling fallbacks
- [ ] Vector embeddings for semantic search
- [ ] Multi-user support with session management
- [ ] Web UI frontend
- [ ] Plugin system for third-party tools
- [ ] Streaming responses
- [ ] Caching layer for repeated queries
- [ ] Multi-language support

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No API key | Set `DEEPSEEK_API_KEY` env or pass `--api-key` |
| Connection failed | Check internet, verify API key, test at `/status` |
| Invalid JSON | Check LLM output format in system prompt |
| Memory growing | Reduce `max_messages` or enable persistence |
| Slow responses | Check network, may be API latency |
| Tool not found | Verify tool is registered in ToolRegistry |
