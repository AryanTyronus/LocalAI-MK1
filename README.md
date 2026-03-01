# LocalAI

LocalAI is a Flask-based local assistant with streaming chat, multi-mode orchestration, structured memory, and reactive tool execution.

## Core Features

- Streaming chat via `/chat/stream` (SSE)
- Multi-mode runtime: `chat`, `coding`, `research`, `agent`
- Registry-driven slash commands (`/help`, `/tool`) with metadata
- Dynamic `/help` UI card (human-facing command/tool list, no raw regex)
- 3-layer memory:
  - short-term conversation
  - structured long-term profile/state
  - vector semantic memory
- Deterministic handlers for:
  - local date/time
  - direct memory recall (name, difficulties)
- Reactive tools (no hidden background agents):
  - `python_executor`
  - `file_reader`
  - `stock_fetcher`
  - `news_fetcher`
  - `weather_fetcher`
  - `indian_market_fetcher`
  - `current_affairs_fetcher`

## UI

- Dark developer-console style layout
- Fixed top status bar + fixed left control sidebar
- Isolated scroll chat feed with anchored composer
- Styled system cards for structured responses (including `/help`)
- Memory and tool-log drawers

## Architecture

- `app.py`: HTTP layer only (request/response/error formatting)
- `core/dependency_container.py`: dependency wiring and tool registration
- `core/generation_pipeline.py`: orchestration (memory, tools, slash commands, generation)
- `core/command_registry.py`: central slash command metadata + handlers
- `core/tool_router.py`: internal trigger matching (backend-only patterns)
- `core/tool_registry.py`: executable tool registry (handler + metadata)
- `memory/`: short-term/long-term/vector memory components
- `tools/`: tool implementations
- `templates/`, `static/`: frontend templates and assets

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start app:
   ```bash
   python app.py
   ```

Server starts on `http://127.0.0.1:8000` or next available port.

## API

### `POST /chat`

Request:
```json
{
  "message": "Hello",
  "mode": "chat",
  "options": {
    "project": "LocalAI",
    "memory_enabled": true,
    "dev_logs": false
  }
}
```

Response:
```json
{
  "response": "...",
  "memory_updated": false
}
```

### `POST /chat/stream`

SSE stream of text chunks.

Special stream messages:
- `__TOKENS__:{...}` final token/runtime payload
- `__SYSTEM_CARD__:{...}` structured system card payload

## Slash Commands

- `/help`: show dynamic command/tool/capability list
- `/tool <request>`: route a request to a matching tool

## Tool Examples

- `weather in London`
- `latest news on AI`
- `stock price for NVDA`
- `indian stock market today`
- `/tool read file app.py`
- `/tool run python: print(2+2)`

## Runtime Notes

- Some tools require internet access.
- FAISS is used when available for vector search.
- `LOCALAI_DEV_MODE=1` uses `FakeModelManager` for lightweight testing.
- If MLX runtime probe fails, LocalAI falls back safely to fake model unless `LOCALAI_REQUIRE_REAL_MODEL=1`.
