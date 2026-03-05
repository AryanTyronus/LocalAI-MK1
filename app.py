"""
Flask application - HTTP layer with ZERO business logic.

This module contains ONLY:
- Request parsing
- Response formatting
- Error handling

NO memory logic, NO prompt building, NO model calls, NO config logic.
All business logic is in AIService.
"""

from flask import Flask, request, jsonify, render_template, Response
import os
import socket
import requests
import re
import json
import subprocess
import time
from urllib.parse import quote_plus
from collections import deque
from datetime import datetime
import threading

try:
    import psutil
except Exception:
    psutil = None

from core.orchestrator import AppOrchestrator
from core.config import DEBUG_MODE, Config
from core.logger import logger
from core.request_policies import (
    needs_live_data as policy_needs_live_data,
    classify_intent,
    classify_response_mode,
    build_context_policy as policy_build_context_policy,
    sanitize_model_reply,
)
from agents.scheduler import AgentScheduler
from agents.tasks import system_monitor_task, daily_summary_task, repository_monitor_task
from prompt_optimization.prompt_tracker import PromptTracker
from prompt_optimization.prompt_optimizer import PromptOptimizer

app = Flask(__name__)

# Global orchestrator
_orchestrator = AppOrchestrator()
_init_lock = threading.Lock()
_synapse_initialized = False
_synapse_ready = False
_ai_service = None
_agent_scheduler = None
_prompt_tracker = PromptTracker()
_prompt_optimizer = PromptOptimizer()

# Web search settings
SERPER_ENDPOINT = "https://google.serper.dev/search"
MAX_SERPER_RESULTS = 3
MAX_CONTEXT_CHARS = 1200
MAX_SYSTEM_LOGS = 20
MAX_REFLECTION_RETRIES = 1
MAX_POLICY_NOTE_CHARS = 240
CLASSIFIER_MAX_TOKENS = 220


# Runtime dashboard state (in-memory, process-local)
_state_lock = threading.Lock()
_model_status = "STANDBY"
_session_metrics = {
    "exchanges": 0,
    "tokens": 0,
    "total_latency_ms": 0.0,
    "latency_count": 0,
    "context_size": 0,
}
_event_logs = deque(maxlen=MAX_SYSTEM_LOGS)
_process = psutil.Process(os.getpid()) if psutil else None
if _process:
    # Prime psutil cpu counters to get meaningful non-zero values later.
    _process.cpu_percent(interval=None)

WHITELISTED_INTENTS = {
    "play_music",
    "play_music_mood",
    "play_specific_song",
    "search_google",
    "search_youtube",
    "open_safari",
    "open_instagram",
    "open_roblox",
    "open_gmail",
    "open_chatgpt",
    "open_github",
    "open_calendar",
    "open_spotify_app",
}


def _load_local_env_file() -> None:
    """
    Lightweight .env loader for local development.

    Reads key=value pairs from .env in the project root and injects them into
    os.environ only when the key is not already set.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Failed to load .env file: {type(e).__name__}: {str(e)}")


_load_local_env_file()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _add_log(message: str) -> None:
    with _state_lock:
        _event_logs.append({
            "timestamp": _now_iso(),
            "message": message,
        })


def _set_model_status(status: str) -> None:
    with _state_lock:
        global _model_status
        _model_status = status


def _estimate_tokens(*texts: str) -> int:
    # Lightweight approximation: word count as token proxy.
    return sum(len((text or "").split()) for text in texts)


def _record_successful_turn(user_message: str, assistant_reply: str, latency_ms: float) -> None:
    with _state_lock:
        _session_metrics["exchanges"] += 1
        _session_metrics["tokens"] += _estimate_tokens(user_message, assistant_reply)
        _session_metrics["total_latency_ms"] += max(0.0, float(latency_ms))
        _session_metrics["latency_count"] += 1
        _session_metrics["context_size"] += 2


def _extract_temperature() -> float | None:
    if not psutil or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False) or {}
    except Exception:
        return None

    preferred_keys = ("coretemp", "k10temp", "cpu-thermal", "soc_thermal", "acpitz")
    for key in preferred_keys:
        entries = temps.get(key) or []
        for entry in entries:
            current = getattr(entry, "current", None)
            if isinstance(current, (int, float)):
                return float(current)

    for entries in temps.values():
        for entry in entries or []:
            current = getattr(entry, "current", None)
            if isinstance(current, (int, float)):
                return float(current)
    return None


_add_log("Backend started")


def get_ai_service():
    """Return pre-initialized AIService."""
    if not _synapse_ready or _ai_service is None:
        raise RuntimeError("SYNAPSE is not initialized.")
    return _ai_service


def initialize_synapse() -> None:
    """
    One-time startup initializer.

    Loads model/service stack, initializes classifier/memory paths, runs warmup
    inference, and marks the process as ready.
    """
    global _synapse_initialized, _synapse_ready, _ai_service, _agent_scheduler

    with _init_lock:
        if _synapse_initialized and _synapse_ready and _ai_service is not None:
            return

        _set_model_status("OFFLINE")
        started = time.perf_counter()
        logger.info("[SYNAPSE] Loading model...")
        _add_log("Loading model")
        try:
            _orchestrator.warm_model()
            _ai_service = _orchestrator.get_ai_service()
            _add_log("Model loaded")

            logger.info("[SYNAPSE] Initializing memory...")
            _add_log("Initializing memory")
            if hasattr(_ai_service, "get_memory_dashboard"):
                _ai_service.get_memory_dashboard()

            logger.info("[SYNAPSE] Initializing intent classifier...")
            _add_log("Initializing intent classifier")
            classify_intent("system status check")
            classify_response_mode("hello", "general_chat")
            _safe_json_extract('{"intent":"normal_chat"}')

            logger.info("[SYNAPSE] Warming up inference...")
            _add_log("Warming up inference")
            _ai_service.generate_response(
                "Reply with exactly: OK",
                mode="chat",
                options={
                    "memory_enabled": False,
                    "include_documents": False,
                    "response_mode": "concise",
                    "reflection_enabled": False,
                },
            )

            _synapse_initialized = True
            _synapse_ready = True
            _set_model_status("STANDBY")

            if _agent_scheduler is None:
                _agent_scheduler = AgentScheduler()
                _agent_scheduler.add_task("system_monitor", 20, system_monitor_task)
                _agent_scheduler.add_task("daily_summary", 3600, daily_summary_task)
                _agent_scheduler.add_task("repository_monitor", 600, repository_monitor_task)
                _agent_scheduler.start()
                _add_log("Background agents started")

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(f"[SYNAPSE] Ready. Startup took {elapsed_ms:.0f} ms.")
            _add_log("SYNAPSE ready")
            print("SYNAPSE ready.")
        except Exception as e:
            _synapse_ready = False
            _synapse_initialized = False
            _ai_service = None
            _set_model_status("ERROR")
            _add_log(f"Initialization error: {type(e).__name__}: {str(e)}")
            logger.exception(f"[SYNAPSE] Initialization failed: {type(e).__name__}: {str(e)}")
            raise


def needs_live_data(prompt: str) -> bool:
    """
    Detect whether a prompt likely needs live web data.

    Uses simple keyword matching as a lightweight trigger.
    """
    return policy_needs_live_data(prompt)


def intent_classifier(prompt: str) -> str:
    """
    Classify request intent before routing.

    Returns one of:
    - general_chat
    - live_data
    - system_query
    - tool_request
    - memory_recall
    - analytical_problem
    """
    return classify_intent(prompt)[0]


def intent_classifier_with_confidence(prompt: str) -> tuple[str, float]:
    """Classify intent and include confidence."""
    return classify_intent(prompt)


def response_mode_classifier(prompt: str, intent: str) -> str:
    """
    Classify desired response style for the final answer.

    Returns one of:
    - concise
    - detailed
    - analytical
    - casual
    - technical
    """
    return classify_response_mode(prompt, intent)[0]


def response_mode_classifier_with_confidence(prompt: str, intent: str) -> tuple[str, float, str]:
    """Classify response mode with confidence and prompt cleanup."""
    return classify_response_mode(prompt, intent)


def search_serper(query: str) -> str:
    """
    Search the web using Serper and return compact context.

    Returns an empty string when search is unavailable or fails.
    """
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        logger.warning("SERPER_API_KEY is not set. Skipping live web search.")
        return ""

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    try:
        response = requests.post(SERPER_ENDPOINT, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.warning(f"Serper request failed: {type(e).__name__}: {str(e)}")
        return ""
    except ValueError:
        logger.warning("Serper response JSON parsing failed.")
        return ""

    organic_results = data.get("organic") or []
    if not isinstance(organic_results, list) or not organic_results:
        return ""

    context_parts = []
    for item in organic_results[:MAX_SERPER_RESULTS]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        link = (item.get("link") or "").strip()
        if not snippet:
            continue
        piece = f"- {title}: {snippet}"
        if link:
            piece += f" ({link})"
        context_parts.append(piece)

    context = "\n".join(context_parts).strip()
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rstrip() + "..."
    return context


def inject_context(original_prompt: str, context: str) -> str:
    """
    Build an augmented prompt that includes external web context.
    """
    if not context:
        return original_prompt
    return (
        "System: Use this web search context to answer the question accurately.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "User:\n"
        f"{original_prompt}"
    )


def _get_system_snapshot() -> dict:
    """
    Collect lightweight runtime snapshot for system_query intent.
    """
    cpu_percent = float(psutil.cpu_percent(interval=None)) if psutil else 0.0
    ram_percent = float(psutil.virtual_memory().percent) if psutil else 0.0
    process_cpu_percent = float(_process.cpu_percent(interval=None)) if _process else 0.0
    temperature = _extract_temperature()
    if temperature is None:
        temperature = 35.0 + (cpu_percent * 0.35)
    with _state_lock:
        status = _model_status
        exchanges = int(_session_metrics["exchanges"])
        tokens = int(_session_metrics["tokens"])
        context_size = int(_session_metrics["context_size"])
        latency_count = int(_session_metrics["latency_count"])
        total_latency_ms = float(_session_metrics["total_latency_ms"])
    avg_latency_ms = (total_latency_ms / latency_count) if latency_count > 0 else 0.0
    return {
        "status": status,
        "cpu_percent": round(cpu_percent, 2),
        "ram_percent": round(ram_percent, 2),
        "process_cpu_percent": round(process_cpu_percent, 2),
        "temperature": round(float(temperature), 2),
        "exchanges": exchanges,
        "tokens": tokens,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "context_size": context_size,
    }


def _format_system_query_response(snapshot: dict) -> str:
    return (
        "System snapshot:\n"
        f"- Status: {snapshot['status']}\n"
        f"- CPU: {snapshot['cpu_percent']}%\n"
        f"- RAM: {snapshot['ram_percent']}%\n"
        f"- Inference Bus: {snapshot['process_cpu_percent']}%\n"
        f"- Thermal: {snapshot['temperature']}°C\n"
        f"- Exchanges: {snapshot['exchanges']}\n"
        f"- Tokens: {snapshot['tokens']}\n"
        f"- Avg Latency: {snapshot['avg_latency_ms']} ms\n"
        f"- Context Size: {snapshot['context_size']}"
    )


def _should_reflect_response(
    user_message: str,
    reply: str,
    intent: str,
    strictness: float = 1.0,
) -> tuple[bool, str]:
    """
    Lightweight quality check to decide whether to regenerate once.
    """
    user_text = (user_message or "").strip()
    answer = (reply or "").strip()
    lowered = answer.lower()

    if not answer:
        return True, "empty_response"
    if lowered in {"[no response]", "no response"}:
        return True, "no_response_placeholder"
    if lowered.startswith("sorry, i couldn't generate"):
        return True, "generation_failure_fallback"

    is_question = "?" in user_text or user_text.lower().split()[:1] in [["what"], ["why"], ["how"], ["when"], ["where"], ["who"]]
    min_words_by_intent = {
        "analytical_problem": 12,
        "technical": 9,
        "live_data": 6,
        "general_chat": 5,
        "memory_recall": 3,
        "tool_request": 4,
    }
    base_min_words = min_words_by_intent.get(intent, 5)
    min_words = max(3, int(base_min_words * max(0.5, strictness)))
    if is_question and len(answer.split()) < min_words:
        return True, "too_short_for_question"

    if intent == "analytical_problem":
        # Analytical answers should usually contain structure.
        if len(answer.split()) < max(10, int(12 * max(0.8, strictness))):
            return True, "analytical_too_brief"
        if not re.search(r"(\d|=|\btherefore\b|\bso\b|\bstep\b)", lowered):
            return True, "analytical_lacks_reasoning_markers"

    # Trivial echo detection
    if answer.lower() == user_text.lower():
        return True, "echoed_user_prompt"

    # Reasoning leakage markers should trigger reflection/sanitization.
    if re.search(r"\b(chain[- ]of[- ]thought|internal reasoning|let me think|reasoning:)\b", lowered):
        return True, "reasoning_leakage_marker"

    return False, "ok"


def _build_reflection_prompt(
    original_prompt: str,
    previous_answer: str,
    reason: str,
    extra_guidance: str = "",
) -> str:
    """
    Build a one-time refinement prompt from the first draft.
    """
    guidance = (extra_guidance or "").strip()
    guidance_block = f"\nOptimization Guidance:\n{guidance}\n" if guidance else ""
    return (
        "System: Improve the assistant answer for correctness and completeness.\n"
        "Do not reveal internal reasoning. Return only the final improved answer.\n\n"
        f"User Question:\n{original_prompt}\n\n"
        f"Previous Draft:\n{previous_answer}\n\n"
        f"Issue Detected:\n{reason}\n\n"
        f"{guidance_block}"
        "Task:\nProvide a better final answer."
    )


def _build_context_policy(prompt: str, intent: str) -> dict:
    """
    Decide which context sources to inject for this turn.

    Rules:
    - analytical_problem (math-like): no web
    - live_data/news: web
    - memory_recall: memory only (disable doc retrieval)
    - mixed (memory + live): combine memory + web, disable docs
    """
    return policy_build_context_policy(prompt, intent)


def _apply_context_policy_options(options: dict, policy: dict) -> dict:
    merged = dict(options or {})
    merged["memory_enabled"] = bool(policy.get("memory_enabled", True))
    merged["include_documents"] = bool(policy.get("include_documents", True))
    return merged


def _build_policy_note(policy: dict) -> str:
    src = ", ".join(policy.get("sources") or ["none"])
    reason = str(policy.get("reason", "unspecified"))
    note = f"Context policy: {reason}. Sources used: {src}."
    if len(note) > MAX_POLICY_NOTE_CHARS:
        note = note[:MAX_POLICY_NOTE_CHARS].rstrip() + "..."
    return note


def _parse_reflection_strictness(value, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.5, min(1.8, parsed))


def _sanitize_param_text(value, max_len: int = 120) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    # Keep alnum/space/common punctuation only.
    text = re.sub(r"[^a-zA-Z0-9\s\-\_\.\,\!\?\'\"\&\(\):]", "", text)
    return text[:max_len].strip()


def _fallback_extract_query(raw_message: str, platform: str) -> str:
    """
    Extract search query from user message when classifier misses parameters.
    """
    text = _sanitize_param_text(raw_message or "", max_len=240)
    if not text:
        return ""
    lowered = text.lower()

    if platform == "youtube":
        patterns = [
            r"(?:search\s+youtube\s+(?:for\s+)?)\s*(.+)$",
            r"(?:youtube\s+(?:search\s+)?(?:for\s+)?)\s*(.+)$",
            r"(?:play\s+on\s+youtube\s+)\s*(.+)$",
        ]
    else:
        patterns = [
            r"(?:search\s+google\s+(?:for\s+)?)\s*(.+)$",
            r"(?:google\s+(?:search\s+)?(?:for\s+)?)\s*(.+)$",
        ]

    for pattern in patterns:
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            candidate = _sanitize_param_text(m.group(1), max_len=180)
            if candidate:
                return candidate

    # Last resort: return original text minus command words.
    stripped = re.sub(r"\b(search|google|youtube|for|on|play|open)\b", "", lowered, flags=re.IGNORECASE)
    return _sanitize_param_text(stripped, max_len=180)


def _is_trivial_query(query: str) -> bool:
    value = (query or "").strip().lower()
    if not value:
        return True
    trivial = {
        "open", "youtube", "google", "search", "play",
        "open youtube", "open google", "search youtube", "search google"
    }
    return value in trivial or len(value) < 2


def _safe_json_extract(raw_text: str) -> dict:
    if not isinstance(raw_text, str):
        return {"intent": "normal_chat"}
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"intent": "normal_chat"}


def _build_intent_classifier_prompt(user_message: str) -> str:
    allowed = ", ".join(sorted(WHITELISTED_INTENTS))
    return (
        "You are an intent classifier. Output STRICT JSON only.\n"
        "No markdown. No extra text.\n\n"
        "Allowed intents:\n"
        f"{allowed}\n\n"
        "If user message matches an allowed intent, return:\n"
        "{\"intent\": \"<intent_name>\", \"parameters\": {...}}\n\n"
        "Otherwise return exactly:\n"
        "{\"intent\": \"normal_chat\"}\n\n"
        "Parameter hints:\n"
        "- search_google: {\"query\": \"...\"}\n"
        "- search_youtube: {\"query\": \"...\"}\n"
        "- play_music_mood: {\"mood\": \"...\"}\n"
        "- play_specific_song: {\"song\": \"...\"}\n\n"
        f"User message:\n{user_message}"
    )


def classify_whitelisted_intent(user_message: str, svc) -> dict:
    # Fast keyword-based matching for greetings and common phrases
    text_lower = user_message.lower().strip()
    
    # Never match greetings - these should always be normal chat
    greetings = ["hello", "hi", "hey", "how are", "good morning", "good evening", "good afternoon"]
    for g in greetings:
        if text_lower == g or text_lower.startswith(g + " ") or text_lower.startswith(g + ","):
            return {"intent": "normal_chat", "parameters": {}}
    
    # Only match explicit tool requests - user must EXPLICITLY ask to open something
    # Check if this is an explicit tool request before calling LLM
    explicit_tool_keywords = [
        "open gmail", "launch gmail", "go to gmail", "check gmail", "start gmail",
        "open chatgpt", "launch chatgpt", "go to chatgpt", "use chatgpt", "start chatgpt",
        "open github", "launch github", "go to github", "start github",
        "open safari", "launch safari", "start safari",
        "open instagram", "launch instagram", "start instagram",
        "open roblox", "launch roblox", "start roblox",
        "open calendar", "launch calendar", "check calendar", "start calendar",
        "open spotify", "launch spotify", "start spotify",
        "search google for", "google search for",
        "search youtube for", "find on youtube",
    ]
    
    is_explicit_tool_request = any(keyword in text_lower for keyword in explicit_tool_keywords)
    
    # If not an explicit tool request, return normal_chat immediately (skip LLM)
    if not is_explicit_tool_request:
        return {"intent": "normal_chat", "parameters": {}}
    
    # Only use LLM for explicit tool requests
    prompt = _build_intent_classifier_prompt(user_message)
    try:
        raw = svc.generate_response(
            prompt,
            mode="chat",
            options={
                "memory_enabled": False,
                "include_documents": False,
                "response_mode": "concise",
                "reflection_enabled": False,
            },
        )
    except Exception:
        return {"intent": "normal_chat", "parameters": {}}

    parsed = _safe_json_extract(raw)
    intent = parsed.get("intent", "normal_chat")
    if not isinstance(intent, str):
        intent = "normal_chat"
    intent = intent.strip()
    if intent not in WHITELISTED_INTENTS:
        return {"intent": "normal_chat", "parameters": {}}
    params = parsed.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    return {"intent": intent, "parameters": params}


def _open_url(url: str) -> bool:
    try:
        subprocess.Popen(["open", url])
        return True
    except Exception:
        return False


def _open_app(app_name: str) -> bool:
    try:
        subprocess.Popen(["open", "-a", app_name])
        return True
    except Exception:
        return False


def open_safari(_params: dict) -> tuple[bool, str]:
    ok = _open_app("Safari")
    return ok, ("Opening Safari." if ok else "Could not open Safari.")


def open_instagram(_params: dict) -> tuple[bool, str]:
    ok = _open_url("https://www.instagram.com")
    return ok, ("Opening Instagram." if ok else "Could not open Instagram.")


def open_roblox(_params: dict) -> tuple[bool, str]:
    ok = _open_url("https://www.roblox.com")
    return ok, ("Opening Roblox." if ok else "Could not open Roblox.")


def open_gmail(_params: dict) -> tuple[bool, str]:
    ok = _open_url("https://mail.google.com")
    return ok, ("Opening Gmail." if ok else "Could not open Gmail.")


def open_chatgpt(_params: dict) -> tuple[bool, str]:
    ok = _open_url("https://chat.openai.com")
    return ok, ("Opening ChatGPT." if ok else "Could not open ChatGPT.")


def open_github(_params: dict) -> tuple[bool, str]:
    ok = _open_url("https://github.com")
    return ok, ("Opening GitHub." if ok else "Could not open GitHub.")


def open_calendar(_params: dict) -> tuple[bool, str]:
    ok = _open_app("Calendar")
    return ok, ("Opening Calendar." if ok else "Could not open Calendar.")


def open_spotify_app(_params: dict) -> tuple[bool, str]:
    ok = _open_app("Spotify")
    return ok, ("Opening Spotify app." if ok else "Could not open Spotify app.")


def search_google(params: dict) -> tuple[bool, str]:
    query = _sanitize_param_text(params.get("query", ""))
    if not query:
        query = _fallback_extract_query(params.get("__raw_message", ""), platform="google")
    if _is_trivial_query(query):
        return False, "Google search query is missing."
    ok = _open_url(f"https://www.google.com/search?q={quote_plus(query)}")
    return ok, (f"Searching Google for '{query}'." if ok else "Could not start Google search.")


def search_youtube(params: dict) -> tuple[bool, str]:
    query = _sanitize_param_text(params.get("query", ""))
    if not query:
        query = _fallback_extract_query(params.get("__raw_message", ""), platform="youtube")
    if _is_trivial_query(query):
        ok = _open_url("https://www.youtube.com")
        return ok, ("Opening YouTube." if ok else "Could not open YouTube.")
    ok = _open_url(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
    return ok, (f"Searching YouTube for '{query}'." if ok else "Could not start YouTube search.")


def play_music(_params: dict) -> tuple[bool, str]:
    ok = _open_url("spotify:home") or _open_url("https://open.spotify.com")
    return ok, ("Playing music on Spotify." if ok else "Could not start music playback.")


def play_music_mood(params: dict) -> tuple[bool, str]:
    mood = _sanitize_param_text(params.get("mood", ""))
    if not mood:
        return False, "Music mood is missing."
    ok = _open_url(f"https://open.spotify.com/search/{quote_plus(mood)}")
    return ok, (f"Playing {mood} music on Spotify." if ok else "Could not open Spotify mood search.")


def play_specific_song(params: dict) -> tuple[bool, str]:
    song = _sanitize_param_text(params.get("song", ""))
    if not song:
        return False, "Song name is missing."
    ok = _open_url(f"https://open.spotify.com/search/{quote_plus(song)}")
    return ok, (f"Playing '{song}' on Spotify." if ok else "Could not open Spotify song search.")


INTENT_TO_FUNCTION = {
    "open_safari": open_safari,
    "open_instagram": open_instagram,
    "open_roblox": open_roblox,
    "open_gmail": open_gmail,
    "open_chatgpt": open_chatgpt,
    "open_github": open_github,
    "open_calendar": open_calendar,
    "open_spotify_app": open_spotify_app,
    "search_google": search_google,
    "search_youtube": search_youtube,
    "play_music": play_music,
    "play_music_mood": play_music_mood,
    "play_specific_song": play_specific_song,
}


def execute_whitelisted_intent(intent_payload: dict) -> tuple[bool, str]:
    intent = intent_payload.get("intent", "normal_chat")
    if intent not in WHITELISTED_INTENTS:
        return False, "normal_chat"
    handler = INTENT_TO_FUNCTION.get(intent)
    if not callable(handler):
        return False, "normal_chat"
    params = intent_payload.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    # Attach raw message for safe fallback extraction when parameters are incomplete.
    raw_message = intent_payload.get("__raw_message", "")
    if isinstance(raw_message, str) and raw_message:
        params = dict(params)
        params["__raw_message"] = raw_message
    return handler(params)


# ===================
# Routes
# ===================

@app.route("/")
def home():
    """Render the landing page."""
    cfg = Config()
    return render_template(
        "landing.html",
        launch_url="/app",
        local_chat_endpoint="/chat",
        api_formats_count=4,  # localai, ollama, openai-compatible, lmstudio
        projects_count=len(getattr(cfg, "ui_projects", []) or []),
    )


@app.route("/app")
def app_page():
    """Render the chat application page."""
    return render_template("index.html", projects=Config().ui_projects)


@app.route("/chat", methods=["POST"])
def chat_api():
    """
    Chat endpoint - thin controller with ZERO business logic.
    
    Expected JSON payload:
        {
            "message": "user message",
            "mode": "chat"  // optional, defaults to "chat"
        }
    
    Returns:
        JSON response with "response" field
    """
    # 1. Parse request
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"response": "Invalid request format."}), 400
    
    user_message = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    options = data.get("options", {})
    
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    # 2. Call AIService (all business logic is here)
    _add_log("Query sent")
    _set_model_status("PROCESSING")
    started_at = datetime.now().timestamp()
    try:
        svc = get_ai_service()

        # --- Whitelisted intent classification + secure execution layer ---
        intent_payload = classify_whitelisted_intent(user_message, svc)
        intent_payload["__raw_message"] = user_message
        if intent_payload.get("intent") in WHITELISTED_INTENTS:
            _add_log(f"Whitelisted intent detected: {intent_payload.get('intent')}")
            executed, confirmation = execute_whitelisted_intent(intent_payload)
            confirmation_text = confirmation if isinstance(confirmation, str) else "Command executed."
            confirmation_text = confirmation_text.strip() or "Command executed."
            latency_ms = (datetime.now().timestamp() - started_at) * 1000.0
            _record_successful_turn(user_message, confirmation_text, latency_ms)
            _set_model_status("STANDBY")
            _add_log(f"Intent execution {'ok' if executed else 'failed'} ({int(latency_ms)} ms)")
            return jsonify({
                "response": confirmation_text,
                "memory_updated": False,
                "intent": intent_payload.get("intent"),
                "intent_payload": intent_payload,
                "executed": bool(executed),
            })

        options = dict(options or {})
        reflection_enabled = bool(options.get("reflection_enabled", True))
        reflection_strictness = _parse_reflection_strictness(options.get("reflection_strictness", 1.0))

        intent, intent_confidence = intent_classifier_with_confidence(user_message)
        _add_log(f"Intent classified: {intent} ({intent_confidence:.2f})")
        response_mode, response_mode_confidence, cleaned_user_message = response_mode_classifier_with_confidence(user_message, intent)
        options["response_mode"] = response_mode
        _add_log(f"Response mode: {response_mode} ({response_mode_confidence:.2f})")
        if cleaned_user_message != user_message:
            _add_log("Style override directive detected and applied")
        policy = _build_context_policy(cleaned_user_message, intent)
        options = _apply_context_policy_options(options, policy)
        _add_log(f"Context sources: {', '.join(policy.get('sources') or ['none'])}")

        if intent == "system_query":
            snapshot = _get_system_snapshot()
            reply = _format_system_query_response(snapshot)
            latency_ms = (datetime.now().timestamp() - started_at) * 1000.0
            _record_successful_turn(user_message, reply, latency_ms)
            _set_model_status("STANDBY")
            _add_log(f"Response received ({int(latency_ms)} ms)")
            return jsonify({
                "response": reply,
                "memory_updated": False,
                "intent": intent,
                "intent_confidence": round(intent_confidence, 3),
                "response_mode": response_mode,
                "response_mode_confidence": round(response_mode_confidence, 3),
                "policy": policy,
            })

        prompt_to_send = cleaned_user_message
        policy_note = _build_policy_note(policy)
        if policy.get("use_web"):
            search_context = search_serper(cleaned_user_message)
            prompt_to_send = inject_context(f"{cleaned_user_message}\n\n[{policy_note}]", search_context)
        else:
            prompt_to_send = f"{cleaned_user_message}\n\n[{policy_note}]"
        
        # Use the new generate_response method
        reply = svc.generate_response(prompt_to_send, mode=mode, options=options)
        reflected = False
        reflection_reason = "ok"
        if intent != "system_query" and reflection_enabled:
            should_retry, reflection_reason = _should_reflect_response(cleaned_user_message, reply, intent, strictness=reflection_strictness)
            if should_retry:
                _add_log(f"Reflection triggered: {reflection_reason}")
                trigger_reason = reflection_reason
                _prompt_tracker.track_event({
                    "type": "reflection_triggered",
                    "intent": intent,
                    "reason": reflection_reason,
                })
                guidance = _prompt_optimizer.get_guidance(reflection_reason) if _prompt_optimizer.is_enabled() else ""
                refinement_prompt = _build_reflection_prompt(
                    cleaned_user_message,
                    reply,
                    reflection_reason,
                    extra_guidance=guidance,
                )
                retries = 0
                while retries < MAX_REFLECTION_RETRIES:
                    retries += 1
                    reflected = True
                    reply = svc.generate_response(refinement_prompt, mode=mode, options=options)
                    should_retry_again, next_reason = _should_reflect_response(cleaned_user_message, reply, intent, strictness=reflection_strictness)
                    if not should_retry_again:
                        reflection_reason = "resolved"
                        break
                    reflection_reason = next_reason
                _prompt_optimizer.learn_from_outcome(trigger_reason, resolved=(reflection_reason == "resolved"))
                _prompt_tracker.track_event({
                    "type": "reflection_result",
                    "intent": intent,
                    "result": reflection_reason,
                    "retries": retries,
                })
                if reflection_reason != "resolved":
                    _add_log(f"Reflection unresolved: {reflection_reason}")
                else:
                    _add_log("Reflection resolved")
            else:
                _add_log("Reflection passed")
        elif not reflection_enabled:
            _add_log("Reflection disabled by request")

        reply = sanitize_model_reply(reply)
        turn_meta = svc.get_last_turn_meta() if hasattr(svc, "get_last_turn_meta") else {}
        latency_ms = (datetime.now().timestamp() - started_at) * 1000.0
        _record_successful_turn(cleaned_user_message, reply, latency_ms)
        _set_model_status("STANDBY")
        _add_log(f"Response received ({int(latency_ms)} ms)")
        
        return jsonify({
            "response": reply,
            "memory_updated": bool(turn_meta.get("memory_updated", False)),
            "intent": intent,
            "intent_confidence": round(intent_confidence, 3),
            "response_mode": response_mode,
            "response_mode_confidence": round(response_mode_confidence, 3),
            "reflected": reflected,
            "reflection_reason": reflection_reason,
            "policy": policy,
        })
    
    except Exception as e:
        import traceback
        logger.error(f"Chat endpoint error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        _set_model_status("ERROR")
        _add_log(f"Error: {type(e).__name__}: {str(e)}")
        
        if DEBUG_MODE:
            return jsonify({
                "response": f"Backend error: {type(e).__name__}: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__
            }), 500
        else:
            return jsonify({"response": "Something went wrong while generating a response."}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    Expected JSON payload:
        {
            "message": "user message",
            "mode": "chat"  // optional, defaults to "chat"
        }
    
    Returns:
        Server-Sent Events stream with "data" field
    """
    # 1. Parse request
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"response": "Invalid request format."}), 400
    
    user_message = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    options = data.get("options", {})
    
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    # 2. Stream response using SSE
    _add_log("Query sent")
    _set_model_status("PROCESSING")
    options = dict(options or {})
    reflect_stream = bool(options.get("reflect_stream", False))
    reflection_enabled = bool(options.get("reflection_enabled", True))
    reflection_strictness = _parse_reflection_strictness(options.get("reflection_strictness", 1.0))

    intent, intent_confidence = intent_classifier_with_confidence(user_message)
    _add_log(f"Intent classified: {intent} ({intent_confidence:.2f})")
    response_mode, response_mode_confidence, cleaned_user_message = response_mode_classifier_with_confidence(user_message, intent)
    options["response_mode"] = response_mode
    _add_log(f"Response mode: {response_mode} ({response_mode_confidence:.2f})")
    if cleaned_user_message != user_message:
        _add_log("Style override directive detected and applied")

    policy = _build_context_policy(cleaned_user_message, intent)
    options = _apply_context_policy_options(options, policy)
    _add_log(f"Context sources: {', '.join(policy.get('sources') or ['none'])}")
    if reflect_stream and reflection_enabled:
        _add_log("Stream reflection enabled (buffered)")
    else:
        _add_log("Reflection skipped for streaming mode")

    if intent == "system_query":
        snapshot = _get_system_snapshot()
        response_text = _format_system_query_response(snapshot)
        latency_ms = 1.0
        _record_successful_turn(cleaned_user_message, response_text, latency_ms)
        _set_model_status("STANDBY")
        _add_log("Response received (1 ms)")
        return Response(f"data: {response_text}\n\n", mimetype="text/event-stream")

    def generate():
        stream_started_at = datetime.now().timestamp()
        stream_reply_parts = []
        try:
            svc = get_ai_service()
            policy_note = _build_policy_note(policy)
            prompt_to_send = f"{cleaned_user_message}\n\n[{policy_note}]"
            if policy.get("use_web"):
                search_context = search_serper(cleaned_user_message)
                prompt_to_send = inject_context(f"{cleaned_user_message}\n\n[{policy_note}]", search_context)

            if reflect_stream and reflection_enabled:
                reply = svc.generate_response(prompt_to_send, mode=mode, options=options)
                should_retry, reason = _should_reflect_response(cleaned_user_message, reply, intent, strictness=reflection_strictness)
                if should_retry:
                    _add_log(f"Reflection triggered: {reason}")
                    _prompt_tracker.track_event({
                        "type": "stream_reflection_triggered",
                        "intent": intent,
                        "reason": reason,
                    })
                    guidance = _prompt_optimizer.get_guidance(reason) if _prompt_optimizer.is_enabled() else ""
                    refinement_prompt = _build_reflection_prompt(
                        cleaned_user_message,
                        reply,
                        reason,
                        extra_guidance=guidance,
                    )
                    reply = svc.generate_response(refinement_prompt, mode=mode, options=options)
                    _prompt_optimizer.learn_from_outcome(reason, resolved=True)
                    _prompt_tracker.track_event({
                        "type": "stream_reflection_result",
                        "intent": intent,
                        "result": "resolved",
                    })
                    _add_log("Reflection completed in buffered stream mode")
                reply = sanitize_model_reply(reply)
                yield f"data: {reply}\n\n"
                latency_ms = (datetime.now().timestamp() - stream_started_at) * 1000.0
                _record_successful_turn(cleaned_user_message, reply, latency_ms)
                _set_model_status("STANDBY")
                _add_log(f"Response received ({int(latency_ms)} ms)")
                return
            
            for chunk in svc.generate_stream(prompt_to_send, mode=mode, options=options):
                if not str(chunk).startswith("__TOKENS__:"):
                    stream_reply_parts.append(str(chunk))
                # Send chunk as SSE data
                yield f"data: {chunk}\n\n"
            final_stream_reply = sanitize_model_reply("".join(stream_reply_parts))
            if final_stream_reply != "".join(stream_reply_parts):
                _add_log("Stream output sanitized")
            latency_ms = (datetime.now().timestamp() - stream_started_at) * 1000.0
            _record_successful_turn(cleaned_user_message, final_stream_reply, latency_ms)
            _set_model_status("STANDBY")
            _add_log(f"Response received ({int(latency_ms)} ms)")
                
        except Exception as e:
            import traceback
            logger.error(f"Streaming endpoint error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            _set_model_status("ERROR")
            _add_log(f"Error: {type(e).__name__}: {str(e)}")
            
            error_msg = str(e) if DEBUG_MODE else "Streaming error occurred."
            yield f"data: __ERROR__:{error_msg}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.route("/system/metrics", methods=["GET"])
def system_metrics():
    """
    Real-time machine metrics for dashboard diagnostics.
    """
    cpu_percent = float(psutil.cpu_percent(interval=None)) if psutil else 0.0
    ram_percent = float(psutil.virtual_memory().percent) if psutil else 0.0
    process_cpu_percent = float(_process.cpu_percent(interval=None)) if _process else 0.0
    temperature = _extract_temperature()
    if temperature is None:
        # Safe fallback estimate when hardware sensors are unavailable.
        temperature = 35.0 + (cpu_percent * 0.35)
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "process_cpu_percent": process_cpu_percent,
        "temperature": round(float(temperature), 2) if temperature is not None else None,
    })


@app.route("/system/status", methods=["GET"])
def system_status():
    """
    Current model status for dashboard core-state display.
    """
    with _state_lock:
        status = _model_status
    return jsonify({"status": status})


@app.route("/session/metrics", methods=["GET"])
def session_metrics():
    """
    In-memory chat-session metrics.
    """
    with _state_lock:
        exchanges = int(_session_metrics["exchanges"])
        tokens = int(_session_metrics["tokens"])
        context_size = int(_session_metrics["context_size"])
        latency_count = int(_session_metrics["latency_count"])
        total_latency_ms = float(_session_metrics["total_latency_ms"])
    avg_latency_ms = (total_latency_ms / latency_count) if latency_count > 0 else 0.0
    return jsonify({
        "exchanges": exchanges,
        "tokens": tokens,
        "avg_latency_ms": avg_latency_ms,
        "context_size": context_size,
    })


@app.route("/system/logs", methods=["GET"])
def system_logs():
    """
    Recent backend activity logs (last N events).
    """
    with _state_lock:
        logs = list(_event_logs)
    return jsonify({"logs": logs})


# ===================
# Utility Functions
# ===================

def _find_free_port(preferred: int) -> int:
    """
    Find a free port starting from preferred.
    
    Args:
        preferred: Preferred port number
        
    Returns:
        Available port number
    """
    for port in range(preferred, preferred + 51):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return 0


# ===================
# Main Entry Point
# ===================

if __name__ == "__main__":
    preferred = int(os.getenv('PORT', '8000'))
    chosen = _find_free_port(preferred)
    
    if chosen != preferred:
        print(f"Port {preferred} unavailable — starting on {chosen} instead.")
    else:
        print(f"Starting server on port {chosen}")

    try:
        initialize_synapse()
    except Exception:
        raise SystemExit(1)

    # Run the Flask app
    app.run(host="0.0.0.0", port=chosen, debug=False)
