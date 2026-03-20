# SYNAPSE Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INPUT                             │
│                   (CLI or Programmatic API)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │          SYNAPSE CORE                      │
        │    (orchestrator.process_query)            │
        └────────────────────────────────────────────┘
                    │        │        │
        ┌───────────┼────────┼────────┼────────────┐
        │           │        │        │            │
        ▼           ▼        ▼        ▼            ▼
    ┌──────┐  ┌────────┐ ┌──────┐ ┌────────┐ ┌────────┐
    │PARSE │  │PREPARE │ │SEND  │ │RECEIVE │ │PARSE   │
    │INPUT │  │MESSAGE │ │TO    │ │RESPONSE│ │OUTPUT  │
    └──────┘  │QUEUE   │ │DeepSeek │      │ └────────┘
        │      └────────┘ └──────┘ └────────┘        │
        │                                             │
        └─────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    Is JSON Action?                      │
    (auto detection)                     │
       /        \                        │
      YES       NO                       │
      │          │                       │
      ▼          └───────────────────────┼─────┐
  ┌──────────┐                           │     │
  │PARSE     │                           ▼     ▼
  │ACTION &  │                     ┌───────────────────┐
  │PARAMETERS│                     │Return Response    │
  └─────────┬┘                     │to User            │
      │                            └────────┬──────────┘
      ▼                                     │
  ┌─────────────────────┐                  │
  │TOOL REGISTRY        │ ◄────────────────┘
  │Has tool in registry?│
  └────────┬────────────┘
      │
   YES│
      ▼
  ┌─────────────────┐
  │EXECUTE TOOL     │
  │(with parameters)│
  └────────┬────────┘
      │
      ▼
  ┌──────────────────┐
  │GET TOOL RESULT   │
  └────────┬─────────┘
      │
      ▼
  ┌──────────────────────────┐
  │SEND RESULT BACK TO LLM   │
  │for contextual response   │
  └────────┬─────────────────┘
      │
      ▼
  ┌──────────────────┐
  │GET FINAL         │
  │RESPONSE          │
  └────────┬─────────┘
      │
      ▼
  ┌──────────────────────────┐
  │UPDATE MEMORY             │
  │(store interaction history)
  └────────┬─────────────────┘
      │
      ▼
  ┌─────────────────────────┐
  │RETURN RESULT TO USER    │
  │(response + metadata)    │
  └─────────────────────────┘
```

## Component Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        SYNAPSE PACKAGE                            │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ CORE                                                        │ │
│  │ ├─ SynapseCore (main orchestrator)                         │ │
│  │ │  ├─ Initialize all subsystems                           │ │
│  │ │  ├─ process_query() - main entry point                 │ │
│  │ │  ├─ Memory management                                   │ │
│  │ │  └─ Connection testing                                  │ │
│  │ └─ CLI interface (main.py)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ LLM MODULE           │  │ TOOLS MODULE         │             │
│  │ ├─ DeepSeekClient    │  │ ├─ ToolRegistry      │             │
│  │ │ ├─ chat()         │  │ │ ├─ register()       │             │
│  │ │ ├─ test_connection()  │  │ │ ├─ execute()     │             │
│  │ │ └─ parse_json()   │  │ │ └─ get_tools_*()   │             │
│  │ │ API: base_url     │  │ │ Tools:              │             │
│  │ │ Model: deepseek-reasoner  │ ├─ open_app      │             │
│  │ └──────────────────────┘  │ ├─ search_web     │             │
│  │                           │ ├─ open_file      │             │
│  │                           │ ├─ read_file      │             │
│  │                           │ ├─ write_file     │             │
│  │                           └──────────────────────┘             │
│  │                                                                │
│  │  ┌─────────────────────────────────────────────────────────┐ │
│  │  │ MEMORY MODULE                                           │ │
│  │  │ ├─ ConversationMemory                                  │ │
│  │  │ │ ├─ add_message()                                     │ │
│  │  │ │ ├─ get_messages() - OpenAI format                    │ │
│  │  │ │ ├─ get_summary()                                     │ │
│  │  │ │ ├─ save/load() - persistence                         │ │
│  │  │ │ └─ clear()                                           │ │
│  │  │ └─ Message dataclass (metadata tracking)               │ │
│  │  └─────────────────────────────────────────────────────────┘ │
│  │                                                                │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow: Complete Query Processing

```
INPUT: User asks "Open Spotify"
  │
  ▼
SYNAPSE.process_query("Open Spotify")
  │
  ├─ Memory.add_message("user", "Open Spotify")
  │
  ├─ DeepSeekClient.chat([all_messages])
  │
  │  Request sent to: https://api.deepseek.com
  │  Model: deepseek-reasoner
  │  Messages: [{"role": "user", "content": "Open Spotify"}]
  │
  │  Response received:
  │  {
  │    "action": "open_app",
  │    "parameters": {"app": "spotify"},
  │    "reasoning": "User wants to open Spotify app"
  │  }
  │
  ├─ JSON detected ✓
  │
  ├─ Extract: action="open_app", params={"app": "spotify"}
  │
  ├─ ToolRegistry.execute("open_app", {"app": "spotify"})
  │   │
  │   └─ Runs: subprocess.run(["open", "-a", "Spotify"])
  │      Returns: "Opened application: Spotify"
  │
  ├─ Send result back to LLM for context
  │
  │  Message: "Action 'open_app' executed with result: Opened application: Spotify
  │            Provide a brief response to user based on this result"
  │
  │  LLM responds: "I've opened Spotify for you. You can now select your
  │                favorite music and start listening!"
  │
  ├─ Memory.add_message("assistant", response, action="open_app")
  │
  ▼
OUTPUT:
{
  "response": "I've opened Spotify for you. You can now select...",
  "thinking": None,
  "action": "open_app",
  "action_result": "Opened application: Spotify",
  "complete": True
}

CLI displays:
  🤖 Response: I've opened Spotify for you...
  🔧 Action Executed: open_app
  📊 Result: Opened application: Spotify
```

## Memory Management Flow

```
Add Message
    │
    ▼
Messages = [msg1, msg2, ..., msgN]
    │
    ├─ Check: len(messages) > max_messages?
    │    │
    │    YES ─► Trim oldest ─► Keep last max_messages
    │    │
    │    NO ─► No change
    │
    ├─ If persist_memory enabled:
    │    │
    │    └─ Save to disk (JSON)
    │        synapse_memory.json
    │
    ▼
Ready for next query
(context available for next LLM call)
```

## Tool Execution Pipeline

```
LLM Response: { "action": "X", "parameters": {...} }
    │
    ▼
Try: Parse as JSON
    │
    ├─ JSON Valid? YES ─► Extract action & params
    │              │
    │              ▼
    │          Tool Registry:
    │          action in registry?
    │          │
    │          YES ─► Execute tool(params)
    │          │        │
    │          │        Try:
    │          │        │
    │          │        ├─ Success ─► Return result
    │          │        ├─ Error ───► Log & return error
    │          │
    │          NO ─► Return "Unknown action"
    │
    └─ JSON Invalid? ─► Treat as regular response
                       (no action execution)
```

## Error Handling Architecture

```
User Query
  │
  ▼
Try: Call LLM
  │
  ├─ Success ──────────────────────────────┐
  │                                         │
  └─ APIConnectionError ────┐              │
     RateLimitError ────┐    │              │
     APIError ─────┐    │    │              │
                   ▼    ▼    ▼              │
              Log error → Return to user    │
              "API Error: ..."              │
                                            │
                                            ▼
                         Try: Parse JSON
                         │
                         ├─ JSON Valid?
                         │  │
                         │  YES ─► Try: Execute tool
                         │  │       │
                         │  │       ├─ Success
                         │  │       ├─ ValueError (unknown tool)
                         │  │       ├─ Exception (exec error)
                         │  │
                         │  NO ─► Treat as text (normal)
                         │
                         ▼
                    Add to Memory
                    Return to user
```

---

## Component Interaction Example

### Scenario: User asks "What's in requirements.txt?"

```
┌─ CLI: prompt("👤 You: ")
│  user_input = "What's in requirements.txt?"
│
└──► SynapseCore.process_query(user_input)
     │
     ├─ Memory.add_message("user", "What's in requirements.txt?")
     │
     ├─ DeepSeekClient.chat([{"role": "user", "content": "..."}])
     │  Returns: {
     │    "action": "read_file",
     │    "parameters": {"path": "requirements.txt"},
     │    "reasoning": "Need to read file to answer question"
     │  }
     │
     ├─ ToolRegistry.execute("read_file", {"path": "requirements.txt"})
     │  │
     │  └─ Reads file: "openai>=1.3.0\npython-dotenv>=1.0.0"
     │     Returns: "openai>=1.3.0\npython-dotenv>=1.0.0"
     │
     ├─ DeepSeekClient.chat([...all_messages..., {
     │    "role": "user",
     │    "content": "Action 'read_file' returned:
     │               openai>=1.3.0
     │               python-dotenv>=1.0.0
     │               
     │               Provide response to user"
     │  }])
     │  Returns: {
     │    "content": "The requirements.txt has two packages:
     │               1. OpenAI SDK (v1.3+)
     │               2. Python-dotenv (v1.0+)"
     │  }
     │
     ├─ Memory.add_message("assistant", response, action="read_file")
     │
     └─ Return to user:
        {
          "response": "The requirements.txt has...",
          "action": "read_file",
          "action_result": "openai>=1.3.0\npython-dotenv>=1.0.0",
          "complete": true
        }

└─ CLI: print_response(result)
   Shows:
   🤖 Response: The requirements.txt has two packages...
   🔧 Action Executed: read_file
   📊 Result: openai>=1.3.0...
```

---

## Deployment Architecture

```
Production Deployment
│
├─ Environment Setup
│  ├─ Python 3.8+
│  ├─ pip install -r synapse/requirements.txt
│  └─ export DEEPSEEK_API_KEY="key"
│
├─ Initialization
│  └─ SynapseCore(api_key, persist_memory=True, debug=False)
│
├─ Running
│  └─ python -m synapse.main  (CLI)
│     OR
│     synapse.process_query(user_input)  (API)
│
├─ Monitoring
│  ├─ Logging to file (logging)
│  ├─ Error tracking
│  └─ Memory persistence (synapse_memory.json)
│
└─ Scaling
   ├─ Increase memory: memory_max_messages
   ├─ Add tools: tools.register()
   └─ Custom prompts: system_prompt parameter
```
