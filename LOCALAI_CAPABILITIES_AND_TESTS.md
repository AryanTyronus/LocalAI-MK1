# SYNAPSE Capabilities and Test Matrix

## Capabilities Snapshot

1. Startup Preload
- SYNAPSE initializes model, memory, classifier, and warmup inference at startup.
- Server starts only after readiness.

2. Chat + Streaming
- `POST /chat` for standard request/response.
- `POST /chat/stream` for SSE token streaming.

3. Intent and Response Control
- Intent classification (`general_chat`, `live_data`, `system_query`, `tool_request`, `memory_recall`, `analytical_problem`).
- Response mode classification (`concise`, `detailed`, `analytical`, `casual`, `technical`).
- Reflection pass with max one retry.

4. Smart Context Injection
- Web context via Serper when policy enables `use_web`.
- Memory/doc toggles by context policy.
- Policy note attached into prompt.

5. Whitelisted Command Execution
- Secure intent whitelist + fixed function map.
- Supports app open, web search, and Spotify actions.
- Query sanitization and fallback extraction for missing search parameters.

6. Runtime Dashboard Data
- `/system/metrics`: CPU, RAM, process CPU, temperature.
- `/system/status`: model status.
- `/session/metrics`: exchanges, tokens, avg latency, context size.
- `/system/logs`: last 20 events.

7. Frontend Runtime Behavior
- `/` landing page and `/app` main UI.
- Voice input supported in UI.
- Voice-triggered auto send path supported in UI.
- Voice output (speech synthesis) for voice-origin prompts after full response completion.

---

## Test Matrix

| ID | Capability | Test Type | Steps | Expected Result | Status |
|---|---|---|---|---|---|
| T01 | Startup preload | Manual | Run `python app.py` | Logs show loading/memory/classifier/warmup and then `SYNAPSE ready.` before serving requests | Not Run |
| T02 | Chat endpoint | API | `curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"hello","mode":"chat"}'` | JSON with non-empty `response` | Not Run |
| T03 | Streaming endpoint | API | `curl -N -X POST http://127.0.0.1:8000/chat/stream -H 'Content-Type: application/json' -d '{"message":"stream test","mode":"chat"}'` | SSE `data:` chunks received | Not Run |
| T04 | Live web search injection | API | Set `SERPER_API_KEY`; ask `latest news today` via `/chat` | Response metadata includes policy with web source; no API-key warning in logs | Not Run |
| T05 | System query route inside chat flow | API | Ask: `show system status and cpu usage` via `/chat` | Response contains status/cpu/ram/inference/thermal fields | Not Run |
| T06 | Reflection pass | API | Ask intentionally vague/low-quality trigger prompt; inspect `/system/logs` | Logs show `Reflection triggered` and resolved/unresolved result | Not Run |
| T07 | Intent whitelist execution | API | Send `open github` via `/chat` | JSON includes `intent: open_github`, `executed: true`, short confirmation response | Not Run |
| T08 | YouTube fallback behavior | API | Send `open youtube` via `/chat` | Opens YouTube homepage (no `search_query=open`) and returns confirmation | Not Run |
| T09 | Metrics endpoint | API | `curl -s http://127.0.0.1:8000/system/metrics` | JSON includes `cpu_percent`, `ram_percent`, `process_cpu_percent`, `temperature` | Not Run |
| T10 | Status endpoint transitions | API | Call `/system/status` while idle and during active request | Idle=`STANDBY`, during generation=`PROCESSING` | Not Run |
| T11 | Session metrics increments | API | Check `/session/metrics`, send two chats, check again | Exchanges and tokens increase; avg latency populated | Not Run |
| T12 | Activity logs | API | `curl -s http://127.0.0.1:8000/system/logs` | Up to 20 timestamped events including query/response/error entries | Not Run |
| T13 | Frontend app page load | UI | Open `http://127.0.0.1:8000/app` | SYNAPSE UI loads without JS errors | Not Run |
| T14 | Voice auto-send | UI | Use mic input and finish speaking | Transcript appears and message sends automatically | Not Run |
| T15 | Voice response speaking mode | UI | Send by voice and await full response | Response is rendered as text; speech starts after full response only | Not Run |

---

## Automated Tests in Repo

1. `tests/test_request_policies.py`
- Validates intent classification behavior.
- Validates response-mode parsing.
- Validates context-policy routing rules.

2. `tests/test_prompt_policies.py`
- Validates prompt policy helpers and constraints.

3. `tests/test_error_handling.py`
- Validates backend error-path behavior and guardrails.

4. `tests/system_validation.py`
- Structural/runtime harness using fake model manager for end-to-end component sanity checks.

---

## Quick Test Commands

```bash
python -m pytest -q tests/test_request_policies.py tests/test_prompt_policies.py tests/test_error_handling.py
```

```bash
curl -s http://127.0.0.1:8000/system/status
curl -s http://127.0.0.1:8000/system/metrics
curl -s http://127.0.0.1:8000/session/metrics
curl -s http://127.0.0.1:8000/system/logs
```

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"open github","mode":"chat"}'
```
