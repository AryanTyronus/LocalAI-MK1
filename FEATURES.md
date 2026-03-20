# SYNAPSE LocalAI - Feature List

## Core Platform
- Flask-based local AI assistant with pre-initialized startup lifecycle.
- One-time warm startup sequence for model/service readiness.
- Environment loading via local `.env` support.
- Dependency-container based orchestration architecture.

## Chat and Response APIs
- `POST /chat` for standard AI responses.
- `POST /chat/stream` for SSE streaming responses.
- Multi-mode generation support (`chat`, `coding`, `research`, `agent`).
- Response metadata support (intent, confidence, response mode, policy, reflection status).

## Routing, Policies, and Prompting
- Intent classification pipeline.
- Response-mode classification pipeline.
- Context policy routing (memory/docs/web toggles).
- Prompt building with layered context assembly.
- Output sanitization and request-policy enforcement.
- Optional reflection/regeneration pass with retry cap.

## Retrieval and Knowledge (RAG)
- Local document ingestion pipeline (including PDF/text/code paths).
- Embeddings + vector store + retriever stack.
- Retrieval context injection into generation flow.
- Knowledge index persistence under `memory/knowledge_index`.

## Memory System
- Short-term conversational memory.
- Long-term memory with SQLite persistence.
- Semantic memory retrieval.
- Memory indexing and retrieval services.
- Memory dashboard hooks from AI service path.

## Tools and External Data
- Live web/news data support through fetcher tools.
- Weather, stock, and Indian market data tool modules.
- Person lookup and current affairs fetchers.
- Safe tool routing and tool registry abstraction.
- Python execution and file reading tool support.

## Safe Command Intents
- Whitelisted intent execution layer for approved actions.
- Supported app/search intents include:
  - `play_music`, `play_music_mood`, `play_specific_song`
  - `search_google`, `search_youtube`
  - `open_safari`, `open_instagram`, `open_roblox`, `open_gmail`
  - `open_chatgpt`, `open_github`, `open_calendar`, `open_spotify_app`

## Planning and Capabilities
- Deterministic planning module.
- Capability registry and executor with parameter/path restrictions.
- Tool execution abstraction through capability layer.

## Background Agents and Automation
- Background scheduler initialized with app startup.
- System monitor task.
- Daily summary task.
- Repository monitor task.

## Prompt Optimization
- Prompt event tracking.
- Prompt optimization guidance layer.
- Reflection-aware prompt adjustments.

## Observability and Runtime Dashboard
- `GET /system/status` model state endpoint.
- `GET /system/metrics` CPU/RAM/process/temperature endpoint.
- `GET /session/metrics` conversation session metrics endpoint.
- `GET /system/logs` bounded backend event logs.
- In-memory runtime metrics (exchanges, tokens, latency, context size).

## Frontend/UI
- Landing page route (`/`).
- App chat interface route (`/app`).
- Template partials for topbar/sidebar/drawer/chat.
- Static JS/CSS assets for UI behavior and styling.

## Testing and Validation Assets
- Policy test suite (`tests/test_request_policies.py`, `tests/test_prompt_policies.py`).
- Error handling tests (`tests/test_error_handling.py`).
- System validation harness (`tests/system_validation.py`).
- Capability and architecture documentation included in repo.
