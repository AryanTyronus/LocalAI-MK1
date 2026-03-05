# SYNAPSE Capabilities and Test Matrix

Last updated: March 5, 2026

## Current Capabilities

1. Model Backend
- Active model: `mlx-community/Qwen3-8B-Instruct`.
- MLX runtime loader through `mlx_lm`.
- Startup preload + warmup before serving traffic.

2. Core Chat APIs
- `POST /chat` standard response path.
- `POST /chat/stream` SSE streaming path.
- Existing response schema preserved.

3. Reasoning and Routing
- Intent classification (`general_chat`, `live_data`, `system_query`, `tool_request`, `memory_recall`, `analytical_problem`).
- Response-mode classification (`concise`, `detailed`, `analytical`, `casual`, `technical`).
- Context policy routing (memory/docs/web toggles).
- Reflection pass with one retry cap.

4. Retrieval (Upgrade 1)
- Local RAG pipeline (`knowledge/*`): loader, embeddings, vector store, retriever.
- Supports markdown/text/PDF/code ingestion path.
- Retrieval context injected into prompt builder path.

5. Long-Term Memory (Upgrade 2)
- SQLite-backed memory store (`memory/memory.sqlite3`).
- Short-term message persistence + long-term fact retrieval.
- Memory retrieval integrated into context assembly.

6. Planning + Safe Tooling (Upgrades 3 and 4)
- Planner (`planning/planner.py`) creates deterministic step plans.
- Capability registry + executor (`capabilities/*`) sanitizes parameters and enforces path restrictions.
- Existing tool execution routes preserved.

7. Background Agents (Upgrade 5)
- Async scheduler (`agents/scheduler.py`) starts at server init.
- Tasks: system monitoring, daily heartbeat, repo monitoring.
- Runs in daemon thread and does not block API routes.

8. Prompt Optimization (Upgrade 6)
- Reflection event tracker (`prompt_optimization/prompt_tracker.py`).
- Reversible prompt guidance templates (`prompt_optimization/prompt_optimizer.py`).
- Reflection flow consumes optional guidance safely.

9. System Metrics and Voice
- `/system/status`, `/system/metrics`, `/session/metrics`, `/system/logs` active.
- Voice support remains in frontend runtime path (unchanged by backend upgrades).

---

## Test Results (Executed)

| ID | Test | Result | Notes |
|---|---|---|---|
| T01 | Startup preload sequence | PASS | `initialize_synapse()` completed; model warmup + classifier init + readiness logged. |
| T02 | `GET /system/status` | PASS | HTTP 200, status `STANDBY`. |
| T03 | `GET /system/metrics` | PASS | HTTP 200 with `cpu_percent`, `ram_percent`, `process_cpu_percent`, `temperature`. |
| T04 | `GET /session/metrics` baseline | PASS | HTTP 200, initial counters at zero before chat calls. |
| T05 | `POST /chat` | PASS | HTTP 200, response payload includes intent/mode/policy/reflection fields. |
| T06 | `POST /chat/stream` SSE | PASS | HTTP 200, stream begins with `data:` and includes `__TOKENS__` footer. |
| T07 | Session metrics increment after chat + stream | PASS | `exchanges` and `tokens` increased after requests. |
| T08 | `GET /system/logs` | PASS | HTTP 200, bounded event list returned (max 20). |
| T09 | Local pytest policy suite | BLOCKED | `pytest` not installed in current Python environments (`python3` and `.venv`). |

---

## Tests Not Re-run in This Pass

1. Live web search integration (`SERPER_API_KEY` dependent).
2. Full UI manual checks (voice input/output interaction).
3. `tests/system_validation.py` legacy harness (contains stale assumptions about old service attributes).

---

## Commands Used (This Update)

```bash
LOCALAI_DEV_MODE=1 python3 - <<'PY'
# Flask test_client validation for:
# /chat, /chat/stream, /system/status, /system/metrics, /session/metrics, /system/logs
PY
```

```bash
python3 -m pytest -q tests/test_request_policies.py tests/test_prompt_policies.py tests/test_error_handling.py
.venv/bin/python -m pytest -q tests/test_request_policies.py tests/test_prompt_policies.py tests/test_error_handling.py
```
