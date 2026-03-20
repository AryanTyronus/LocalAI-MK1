# SYNAPSE - Modular AI Assistant with DeepSeek

SYNAPSE is a production-quality AI assistant that integrates DeepSeek's reasoning engine with a modular, extensible architecture.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r synapse/requirements.txt
```

### 2. Set API Key

Either export the environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or you'll be prompted for it when running the CLI.

### 3. Run SYNAPSE

```bash
python -m synapse.main
```

## Architecture

```
synapse/
├── core.py                 # Central orchestrator (SynapseCore)
├── llm/
│   └── deepseek_client.py # OpenAI-compatible DeepSeek wrapper
├── tools/
│   └── tool_registry.py   # Tool system with manual function-calling
├── memory/
│   └── memory.py          # Conversation history & context
└── main.py               # CLI interface
```

## Core Components

### 1. DeepSeekClient (`llm/deepseek_client.py`)

OpenAI-compatible wrapper for DeepSeek API:
- Uses `deepseek-reasoner` model for advanced reasoning
- Handles API errors gracefully
- Parses structured JSON responses
- Optional thinking/reasoning extraction

```python
from synapse.llm import DeepSeekClient

client = DeepSeekClient(api_key="your-key")
response = client.chat([{"role": "user", "content": "Hello"}])
print(response.content)
```

### 2. SynapseCore (`core.py`)

Central orchestrator that:
- Manages user input flow
- Communicates with LLM
- Executes tools based on LLM decisions
- Maintains conversation memory

```python
from synapse import SynapseCore

synapse = SynapseCore(api_key="your-key")
result = synapse.process_query("Open Spotify and search for jazz music")
print(result["response"])
```

### 3. Tool Registry (`tools/tool_registry.py`)

Manual function-calling system (DeepSeek doesn't support native function calling):
- LLM returns structured JSON with `action` and `parameters`
- SYNAPSE parses and executes the tool
- Sends result back to LLM for final response

**Built-in Tools:**
- `open_app` - Open applications or URLs
- `search_web` - Search the web
- `open_file` - Open files
- `read_file` - Read file contents
- `write_file` - Write to files

**Adding New Tools:**
```python
def my_tool(param1, param2):
    return f"Result: {param1} {param2}"

synapse.tools.register(
    "my_tool",
    my_tool,
    "Description of what this tool does",
    {"param1": "Description", "param2": "Description"}
)
```

### 4. Conversation Memory (`memory/memory.py`)

Lightweight memory management:
- Stores last N interactions
- Optional disk persistence
- Preserves context for multi-turn conversations

```python
from synapse.memory import ConversationMemory

memory = ConversationMemory(max_messages=20, persist_path="memory.json")
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi there!")
print(memory.get_messages())  # OpenAI-compatible format
```

## CLI Usage

### Start Interactive Mode
```bash
python -m synapse.main
```

### Available Commands

| Command | Purpose |
|---------|---------|
| `/help` | Show help menu |
| `/memory` | Display conversation summary |
| `/clear` | Clear conversation history |
| `/export` | Export memory to JSON |
| `/status` | Test API connection |
| `/exit` | Exit SYNAPSE |

### Example Session

```
👤 You: Open Spotify

⏳ Processing...

💭 Thinking:
The user wants to open Spotify. I should use the open_app tool.

🤖 Response:
I've opened Spotify for you. You can now play your favorite music!

🔧 Action Executed: open_app
📊 Result: Opened application: Spotify
```

## Advanced Usage

### Custom System Prompt

```bash
python -m synapse.main --system-prompt custom_prompt.txt
```

### Enable Debug Logging

```bash
python -m synapse.main --debug
```

### Persist Memory

```bash
python -m synapse.main --persist
```

### Programmatic Usage

```python
from synapse import SynapseCore

synapse = SynapseCore(
    api_key="your-key",
    system_prompt="You are a helpful assistant specialized in...",
    memory_max_messages=20,
    persist_memory=True,
    debug=True
)

# Process queries
result = synapse.process_query("What's the weather like?")
print(result["response"])
print(result["thinking"])  # Optional reasoning
print(result["action"])     # Tool executed
print(result["action_result"])  # Tool result

# Inspect memory
summary = synapse.get_memory_summary()
print(summary)

# Clear on demand
synapse.clear_memory()
```

## Configuration

Set environment variables to customize behavior:

```bash
export DEEPSEEK_API_KEY="your-key"
export SYNAPSE_DEBUG=true
export SYNAPSE_MAX_MESSAGES=30
```

## Error Handling

SYNAPSE handles:
- ✓ API connection failures → graceful error messages
- ✓ Invalid JSON responses → fallback to treating as text
- ✓ Tool execution errors → communicates failure to user
- ✓ Missing tools → clear error messages
- ✓ Rate limiting → proper exception handling

## Extending SYNAPSE

### Add a Custom Tool

```python
from synapse import SynapseCore

synapse = SynapseCore(api_key="key")

def send_email(recipient, subject, body):
    # Implementation here
    return f"Email sent to {recipient}"

synapse.tools.register(
    "send_email",
    send_email,
    "Send an email to a recipient",
    {
        "recipient": "Email address",
        "subject": "Email subject",
        "body": "Email body"
    }
)
```

### Custom System Prompt

Create a file `my_prompt.txt`:
```
You are SYNAPSE, a specialized assistant for software engineering.
You help with code reviews, debugging, and architecture decisions.
```

Then run:
```bash
python -m synapse.main --system-prompt my_prompt.txt
```

## Performance Notes

- **Memory:** ~50MB for full initialization
- **Latency:** 1-5 seconds per query (depends on DeepSeek API)
- **Token Budget:** Optimized for deep reasoning without excessive context
- **Scalability:** Designed for single-user CLI, can be extended with async/scaling

## Troubleshooting

### "No API key provided"
Set `DEEPSEEK_API_KEY` environment variable or pass `--api-key` flag.

### "Connection failed"
- Check internet connectivity
- Verify API key is valid
- Try `/status` command to test connection

### "Invalid JSON from LLM"
SYNAPSE falls back to treating response as regular text. This is expected for non-action responses.

### Memory growing too large
Configure `max_messages` parameter or use `python -m synapse.main` with custom settings.

## Technical Details

- **LLM Model:** `deepseek-reasoner` (supports extended thinking)
- **SDK:** OpenAI Python SDK v1.3+
- **Memory:** In-memory with optional JSON persistence
- **Threading:** Single-threaded (can be adapted for async)
- **Error Handling:** Full exception propagation with logging

## Future Enhancements

- [ ] Async/streaming responses
- [ ] Multi-turn tool chains
- [ ] Function calling fallbacks
- [ ] Web UI frontend
- [ ] Multi-user support
- [ ] Vector embeddings for semantic memory
- [ ] Plugin system for third-party tools
- [ ] Batch processing mode

## License

This project is provided as-is for educational and production use.

## Support

For issues or questions, refer to the code comments and docstrings throughout the codebase.
