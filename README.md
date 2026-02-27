# LocalAI (Flask + Local Model + Reactive Tools)

LocalAI is a Flask-based local assistant with:
- Streaming chat responses
- Multi-mode orchestration (`chat`, `coding`, `research`, `agent`)
- 3-layer memory (short-term, structured long-term, vector memory)
- Reactive tool execution (no hidden background agents)
- Modern UI with sidebar controls and memory/tool drawers

## Features

- Chat + streaming via `/chat` and `/chat/stream`
- Project selector, mode selector, memory toggle, dev logs toggle
- Token usage footer in streaming responses
- Memory update indicator per turn
- Deterministic date/time responses from local system clock
- Structured memory recall for personal facts (name, difficulties, etc.)
- Tool integrations:
  - `python_executor`
  - `file_reader`
  - `stock_fetcher`
  - `news_fetcher`
  - `weather_fetcher`
  - `indian_market_fetcher`
  - `current_affairs_fetcher` (live current-affairs fact path)

## Architecture (High-Level)

- `app.py`: Flask HTTP layer only (routes, request parsing, responses)
- `core/orchestrator.py`: app startup lifecycle
- `core/dependency_container.py`: dependency wiring
- `core/generation_pipeline.py`: orchestration logic (context, tools, memory, generation)
- `memory/`: structured + vector memory systems
- `tools/`: reactive tool implementations
- `templates/` + `static/`: Flask Jinja frontend

## Quick Start

1. Create and activate virtual env
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python app.py
   ```

Server starts at `http://127.0.0.1:8000` (or next available port).

## API

### `POST /chat`
Body:
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
Streams Server-Sent Events. Final token payload is emitted as:
`__TOKENS__:{...}`

## Tool Usage Examples

- `latest news on artificial intelligence`
- `weather in Mumbai`
- `indian stock market today`
- `nse stock reliance`
- `who is the president of america?`
- `/tool read file app.py`
- `/tool run python: print(2+2)`

## Notes

- News/weather/market/current-affairs tools require internet access.
- Vector memory uses FAISS if available.
- In `LOCALAI_DEV_MODE=1`, a fake model manager is used for lightweight testing.
