# SYNAPSE LocalAI

SYNAPSE is a Flask-based LocalAI assistant with startup preloading, streaming chat, memory, intent routing, tool execution, and live dashboard telemetry.

## Current Capabilities

- Eager startup initialization (`initialize_synapse`) before server begins accepting traffic
- Main chat endpoint and SSE streaming endpoint
- Smart prompt routing with:
  - intent classification
  - response-mode classification
  - context policy selection
  - optional reflection/regeneration pass
- Live web search context injection using Serper (`SERPER_API_KEY`)
- Whitelisted command-intent execution layer (safe mapping only)
- Runtime system dashboard APIs:
  - `/system/metrics`
  - `/system/status`
  - `/session/metrics`
  - `/system/logs`
- UI pages:
  - `/` landing page
  - `/app` SYNAPSE chat interface

## Architecture

- `app.py`: HTTP routes, orchestration hooks, runtime state, startup initialization
- `core/`: orchestration, prompt policies, model adapters, tool routing, config
- `memory/`: short-term, structured, and semantic memory layers
- `tools/`: fetchers and utility tools
- `templates/`, `static/`: landing page and web app UI
- `tests/`: policy and error-handling test modules

## Requirements

- Python 3.10+
- macOS for app-intent handlers (`open -a ...`) used by whitelisted intents
- Optional API key for live web search:
  - `SERPER_API_KEY`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:

```env
SERPER_API_KEY=your_serper_key_here
PORT=8000
```

## Run

```bash
python app.py
```

Expected startup sequence:

- `[SYNAPSE] Loading model...`
- `[SYNAPSE] Initializing memory...`
- `[SYNAPSE] Initializing intent classifier...`
- `[SYNAPSE] Warming up inference...`
- `SYNAPSE ready.`

## API

### `POST /chat`

Request:

```json
{
  "message": "What is the latest AI news today?",
  "mode": "chat",
  "options": {
    "reflection_enabled": true,
    "reflection_strictness": 1.0
  }
}
```

Response fields include:

- `response`
- `memory_updated`
- `intent`, `intent_confidence`
- `response_mode`, `response_mode_confidence`
- `reflected`, `reflection_reason`
- `policy`

### `POST /chat/stream`

SSE stream of generated chunks.

### `GET /system/metrics`

Returns CPU, RAM, process CPU, and temperature estimate/sensor value.

### `GET /system/status`

Returns model state: `STANDBY`, `PROCESSING`, `OFFLINE`, or `ERROR`.

### `GET /session/metrics`

Returns exchanges, tokens, average latency, and context size.

### `GET /system/logs`

Returns last 20 backend activity log events.

## Whitelisted Intents

- `play_music`
- `play_music_mood`
- `play_specific_song`
- `search_google`
- `search_youtube`
- `open_safari`
- `open_instagram`
- `open_roblox`
- `open_gmail`
- `open_chatgpt`
- `open_github`
- `open_calendar`
- `open_spotify_app`

If classification output is outside whitelist, request falls back to normal chat flow.

## Testing

Run fast policy tests:

```bash
python -m pytest -q tests/test_request_policies.py tests/test_prompt_policies.py
```

Run error handling tests:

```bash
python -m pytest -q tests/test_error_handling.py
```

For full capability verification, use:

- [LOCALAI_CAPABILITIES_AND_TESTS.md](/Users/aryandas/Desktop/My%20python%20AI/LocalAI/LOCALAI_CAPABILITIES_AND_TESTS.md)
