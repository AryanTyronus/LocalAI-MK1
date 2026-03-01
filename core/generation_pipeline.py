"""
GenerationPipeline - Core generation orchestration.

This class contains ALL generation logic:
- Message to short-term memory
- Profile facts extraction
- Rolling summary triggers
- Semantic memory retrieval
- Document chunk retrieval
- Token budget enforcement (Phase 2)
- Context compression (Phase 4)
- Prompt building
- Model generation
- Response to memory
- Memory persistence
- Fault tolerance and debug hooks (Phase 6)

AIService should only delegate to this class.
"""

import time
import json
import re
from datetime import datetime
from collections import defaultdict
from typing import Any, Generator, Dict, List, Optional

from core.config import Config, DEBUG_MODE
from core.command_registry import CommandRegistry
from core.logger import logger
from core.mode_controller import ModeController
from core.prompt_builder import PromptBuilder
from core.tool_registry import ToolRegistry
from core.tool_router import ToolRouter, get_tool_router
from core.intent_classifier import IntentClassifier
from core.token_budget import TokenBudgetManager
from memory.memory_manager import MemoryManager
from retrieval.document_manager import DocumentManager


class GenerationPipeline:
    """
    GenerationPipeline handles all generation logic.
    
    Responsibilities:
    - Add message to short-term memory
    - Extract structured profile facts
    - Trigger rolling summary if needed
    - Retrieve semantic memory
    - Retrieve document chunks (if enabled in mode)
    - Enforce token budget (Phase 2)
    - Build prompt via PromptBuilder
    - Call ModelAdapter.generate() or generate_stream()
    - Add assistant response to memory
    - Persist memory if needed
    """
    SYSTEM_CARD_PREFIX = "__SYSTEM_CARD__:"
    
    def __init__(
        self,
        config: Config,
        memory_manager: MemoryManager,
        model_adapter,
        prompt_builder: PromptBuilder,
        mode_controller: ModeController,
        document_manager: DocumentManager = None,
        tool_registry: ToolRegistry = None,
        token_budget_manager: TokenBudgetManager = None,
        tool_router: ToolRouter = None
    ):
        """
        Initialize GenerationPipeline with all dependencies.
        
        Args:
            config: Config instance for settings
            memory_manager: MemoryManager instance for 4-layer memory
            model_adapter: BaseModelInterface adapter for LLM generation
            prompt_builder: PromptBuilder instance for prompt construction
            mode_controller: ModeController instance for mode management
            document_manager: DocumentManager instance for document retrieval
            tool_registry: ToolRegistry instance for tool execution
            token_budget_manager: TokenBudgetManager for context budgeting (Phase 2)
            tool_router: ToolRouter for tool intent detection (Phase 7)
        """
        self._config = config
        self._memory_manager = memory_manager
        self._model_adapter = model_adapter
        self._prompt_builder = prompt_builder
        self._mode_controller = mode_controller
        self._document_manager = document_manager
        self._tool_registry = tool_registry
        
        # Phase 7: ToolRouter for intent detection
        self._tool_router = tool_router or get_tool_router()
        self._tool_router.enable()
        
        # Phase 2: TokenBudgetManager for context enforcement
        self._token_budget_manager = token_budget_manager or TokenBudgetManager(config)
        self._last_turn_meta = {"memory_updated": False}
        
        # Memory triggers for semantic memory
        self._memory_triggers = [
            "my name is",
            "i was born",
            "my birthday is",
            "i like",
            "i prefer",
            "my favorite",
            "i struggle",
            "i enjoy",
            "i love",
            "i hate",
            "my goal",
            "i want"
        ]
        
        # Study keywords for document retrieval heuristics
        self._study_keywords = [
            "physics", "math", "jee", "derive",
            "equation", "numerical", "solve",
            "formula", "calculate", "explain"
        ]

        self._register_builtin_commands()
        
        logger.info("GenerationPipeline initialized")
    
    # ===================
    # Main Generation Methods
    # ===================

    def _register_builtin_commands(self) -> None:
        """Register built-in slash commands in the central registry."""
        CommandRegistry.register_command(
            name="/help",
            description="Show all registered commands, tool triggers, and system capabilities.",
            category="system",
            usage="/help",
            handler=self._handle_help_command,
            store_in_memory=False,
        )
        CommandRegistry.register_command(
            name="/tool",
            description="Execute a tool by natural-language trigger routing.",
            category="tools",
            usage="/tool weather in mumbai",
            handler=self._handle_tool_command,
            store_in_memory=True,
        )

    def _handle_help_command(self, _context: Dict[str, Any], _args: str) -> Dict[str, Any]:
        """Build a dynamic help card payload from live registries."""
        payload = self._build_help_payload()
        return {"system_card": payload}

    def _handle_tool_command(self, context: Dict[str, Any], args: str) -> Dict[str, Any]:
        """Slash command wrapper for tool execution."""
        tool_prompt = (args or "").strip()
        usage = "Usage: /tool <tool request>. Example: /tool weather in Tokyo"
        if not tool_prompt:
            return {"text": usage}

        mode_obj = context.get("mode_obj")
        reply = self._execute_tool_if_requested(f"/tool {tool_prompt}", mode_obj)
        if reply is None:
            return {
                "text": "No matching tool trigger was found for that request. Use /help to see available tool examples."
            }
        return {"text": reply}

    def _build_help_payload(self) -> Dict[str, Any]:
        """Build grouped help payload with commands, tool triggers, and capabilities."""
        command_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for cmd in CommandRegistry.list_commands():
            row = {
                "name": cmd.name,
                "description": cmd.description,
            }
            if cmd.usage:
                row["usage"] = cmd.usage
            command_groups[cmd.category].append(row)

        trigger_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        available_tool_names = {tool.get("name") for tool in self._tool_registry.list_tools()} if self._tool_registry else set()
        for trigger in self._tool_router.get_help_triggers(available_tool_names=available_tool_names):
            trigger_groups[trigger.get("category", "tools")].append(
                {
                    "name": trigger.get("name", ""),
                    "usage": trigger.get("usage", ""),
                    "description": trigger.get("description", ""),
                    "tool_name": trigger.get("tool_name", ""),
                }
            )

        capability_groups = self._build_system_capabilities()
        sections = [
            {
                "kind": "commands",
                "title": "Slash Commands",
                "groups": [
                    {"category": category, "items": sorted(items, key=lambda x: x["name"])}
                    for category, items in sorted(command_groups.items(), key=lambda kv: kv[0])
                ],
            },
            {
                "kind": "tool_triggers",
                "title": "Tool Triggers",
                "groups": [
                    {"category": category, "items": sorted(items, key=lambda x: x["tool_name"])}
                    for category, items in sorted(trigger_groups.items(), key=lambda kv: kv[0])
                ],
            },
            {
                "kind": "capabilities",
                "title": "System Capabilities",
                "groups": [
                    {"category": category, "items": items}
                    for category, items in sorted(capability_groups.items(), key=lambda kv: kv[0])
                ],
            },
        ]

        return {
            "type": "help",
            "title": "LocalAI Help",
            "generated_at": datetime.now().astimezone().isoformat(),
            "sections": sections,
        }

    def _build_system_capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover runtime capabilities from config and registries."""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        mode_names = sorted(getattr(self._mode_controller, "_modes", {}).keys())
        groups["modes"].append({"name": "available_modes", "value": ", ".join(mode_names) or "none"})
        groups["diagnostics"].append({"name": "streaming", "value": "enabled"})
        groups["diagnostics"].append({"name": "tool_router", "value": "enabled" if self._tool_router.is_enabled else "disabled"})

        groups["memory"].append({"name": "short_term", "value": "enabled" if self._config.short_term_enabled else "disabled"})
        groups["memory"].append({"name": "rolling_summary", "value": "enabled" if self._config.rolling_summary_enabled else "disabled"})
        groups["memory"].append({"name": "semantic", "value": "enabled" if self._config.semantic_enabled else "disabled"})
        groups["memory"].append({"name": "structured", "value": "enabled" if self._config.structured_enabled else "disabled"})

        tool_count = len(self._tool_registry.list_tools()) if self._tool_registry else 0
        trigger_count = len(self._tool_router.get_trigger_definitions())
        groups["tools"].append({"name": "registered_tools", "value": str(tool_count)})
        groups["tools"].append({"name": "trigger_patterns", "value": str(trigger_count)})

        doc_count = 0
        if self._document_manager:
            try:
                doc_count = len(self._document_manager.list_documents())
            except Exception:
                doc_count = 0
        groups["retrieval"].append({"name": "loaded_documents", "value": str(doc_count)})

        return groups

    def _execute_slash_command(
        self,
        user_message: str,
        mode_obj,
        runtime: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a slash command if registered and return formatted response details."""
        text = (user_message or "").strip()
        if not text.startswith("/"):
            return None

        parts = text.split(None, 1)
        command_name = parts[0].lower()
        command_args = parts[1] if len(parts) > 1 else ""
        command = CommandRegistry.get_command(command_name)
        if not command:
            return None

        context = {"mode_obj": mode_obj, "runtime": runtime}
        raw_result = command.handler(context, command_args) or {}
        response_text = str(raw_result.get("text", "")).strip()
        system_card = raw_result.get("system_card")
        if system_card:
            response_text = f"{self.SYSTEM_CARD_PREFIX}{json.dumps(system_card, ensure_ascii=True)}"

        return {
            "command": command,
            "response_text": response_text,
        }

    def _resolve_mode(self, user_message: str, mode: str):
        """Resolve mode, including auto intent routing."""
        resolved_mode = mode
        if not resolved_mode or resolved_mode == 'auto':
            inferred, conf = IntentClassifier.classify(user_message)
            logger.info(f"Auto-routed intent: {inferred} (conf={conf:.2f})")
            resolved_mode = inferred
        return resolved_mode, self._mode_controller.get_mode(resolved_mode)

    def _resolve_runtime_options(self, options: Optional[Dict]) -> Dict:
        """Normalize runtime request options from UI/API."""
        opts = options or {}
        memory_enabled = bool(opts.get("memory_enabled", True))
        dev_logs = bool(opts.get("dev_logs", False))
        project = str(opts.get("project", "LocalAI")).strip() or "LocalAI"
        return {
            "memory_enabled": memory_enabled,
            "dev_logs": dev_logs,
            "project": project,
        }

    def _prepare_generation_context(self, user_message: str, mode: str, options: Optional[Dict] = None, stream: bool = False) -> Dict:
        """
        Shared context and prompt preparation for both generate() and run_stream().
        """
        runtime = self._resolve_runtime_options(options)
        mode, mode_obj = self._resolve_mode(user_message, mode)

        memory_updated = False
        if runtime["memory_enabled"] and self._is_personal_message(user_message):
            self._memory_manager.clear_short_term()

        if runtime["memory_enabled"]:
            self._memory_manager.add_short_term_message('user', user_message)
            memory_updated = bool(self._memory_manager.extract_profile_facts(user_message)) or memory_updated
            self._memory_manager.save_all()

        if runtime["memory_enabled"] and self._memory_manager.maybe_create_summary():
            logger.info("Created rolling summary from short-term memory")

        user_lower = user_message.lower()
        if runtime["memory_enabled"] and any(trigger in user_lower for trigger in self._memory_triggers):
            self._memory_manager.add_semantic_memory(user_message)
            self._memory_manager.save_all()
            memory_updated = True

        now_local = datetime.now().astimezone()
        name = self._extract_name()
        birth_year = self._extract_birth_year()
        current_year = now_local.year
        age = self._calculate_age(birth_year)
        target_year = self._extract_target_year(user_lower)
        future_age = self._calculate_age(birth_year, target_year) if birth_year and target_year else None

        docs = []
        should_run_doc_retrieval = self._should_retrieve_documents(user_message) and not self._is_tool_request(user_message)
        if should_run_doc_retrieval:
            try:
                active_doc = self._document_manager.active_doc_id if self._document_manager else None
                if self._document_manager:
                    docs = self._document_manager.search(
                        user_message,
                        top_k=5,
                        doc_id=active_doc
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Document retrieval failed: {e}")
                docs = []

        if runtime["memory_enabled"]:
            full_context = self._memory_manager.build_full_context(user_message)
        else:
            full_context = {'structured': '', 'rolling_summary': '', 'semantic': '', 'short_term': ''}

        retrieved_block = ""
        if docs:
            parts = ["=== Retrieved Document Context ==="]
            for res in docs:
                chunk = res.get('chunk', {})
                meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                parts.append(f"{meta}\n{chunk.get('text')}")
            retrieved_block = "\n\n".join(parts)

        include_structured = runtime["memory_enabled"]
        include_rolling = runtime["memory_enabled"]
        include_semantic = runtime["memory_enabled"] and self._should_use_memory_retrieval(user_message, mode_obj)

        memory_profile = mode_obj.memory_profile
        if memory_profile == 'short_term_minimal':
            include_structured = False
            include_rolling = False
            include_semantic = False
        elif memory_profile == 'short_term_heavy':
            include_rolling = False
            include_semantic = False

        system_message = self._build_system_message(
            mode_obj, name, birth_year, age, current_year, future_age, runtime["project"], now_local
        )

        structured_mem = full_context.get('structured', '') if include_structured else ''
        rolling_summary_mem = full_context.get('rolling_summary', '') if include_rolling else ''
        semantic_mem = full_context.get('semantic', '') if include_semantic else ''
        short_term_conv = full_context.get('short_term', '')

        trimmed_context, budget, trimming_log = self._token_budget_manager.get_trimmed_context(
            system_prompt=system_message,
            structured_memory=structured_mem,
            rolling_summary=rolling_summary_mem,
            semantic_memories=semantic_mem,
            documents=retrieved_block,
            short_term=short_term_conv,
            user_query=user_message
        )

        if trimming_log.get('trimmed_blocks'):
            log_prefix = "Token budget (stream)" if stream else "Token budget"
            logger.info(f"{log_prefix}: {budget.total} tokens, trimming {len(trimming_log['trimmed_blocks'])} blocks")
            for tb in trimming_log['trimmed_blocks']:
                logger.info(f"  - {tb['name']}: {tb['original_tokens']} -> {tb['remaining_tokens']} tokens")

        built = self._prompt_builder.build(
            system_prompt=trimmed_context.get('system_prompt', system_message),
            retrieved_docs=docs,
            memories={
                'structured': trimmed_context.get('structured', ''),
                'rolling_summary': trimmed_context.get('rolling_summary', ''),
                'semantic': trimmed_context.get('semantic', '')
            },
            recent_conv=trimmed_context.get('short_term', ''),
            user_message=user_message,
        )

        if DEBUG_MODE:
            dbg_suffix = " (stream)" if stream else ""
            logger.debug(f"[DEBUG] Full prompt context{dbg_suffix}:\n{built['system']}\n\n{built['user']}")

        prompt = self._model_adapter.format_chat(
            system=built['system'],
            user=built['user'],
            history=None
        )

        return {
            'mode': mode,
            'mode_obj': mode_obj,
            'prompt': prompt,
            'runtime': runtime,
            'memory_updated': memory_updated,
        }

    def _is_tool_request(self, user_message: str) -> bool:
        """Detect explicit tool invocation requests."""
        text = (user_message or "").strip()
        if not text:
            return False
        lowered = text.lower()

        # Guard against false positives on personal-memory statements.
        if re.search(
            r"\b(my name is|i am|i'm|i struggle|i have trouble|i have difficulty|i find .+ difficult|my weak subjects|i like|i prefer)\b",
            lowered,
        ):
            return False

        if lowered.startswith("/tool "):
            return True
        if re.search(r"\b(run|execute)\s+python\b", lowered):
            return True
        if re.search(r"\b(read|show)\s+file\b", lowered):
            return True
        if re.search(r"\b(stock|quote|price|ticker)\b", lowered):
            return True
        detected = self._tool_router.detect_intent(text)
        return detected.tool_name != ""

    def _should_use_memory_retrieval(self, user_message: str, mode_obj) -> bool:
        """
        Memory retrieval decision layer:
        avoid unnecessary semantic retrieval for greetings or explicit tool calls.
        """
        text = (user_message or "").strip().lower()
        if not text:
            return False
        if self._is_tool_request(user_message):
            return False
        if text in {"hi", "hello", "hey", "ok", "thanks"}:
            return False
        if len(text.split()) <= 2 and mode_obj.name != "research":
            return False
        return True

    def _format_tool_response(self, tool_name: str, result: Dict) -> str:
        """
        Format tool outputs for chat-safe return.
        """
        if tool_name == "current_affairs_fetcher" and isinstance(result, dict):
            if not result.get("ok"):
                return f"Current-affairs fetch failed: {result.get('error', 'unknown error')}"
            fact = result.get("fact", "").strip()
            source = result.get("source", "").strip()
            if source:
                return f"{fact}\nSource: {source}"
            return fact or "Live fact retrieved."

        if tool_name == "indian_market_fetcher" and isinstance(result, dict):
            if not result.get("ok"):
                return f"Indian market fetch failed: {result.get('error', 'unknown error')}"

            def _fmt_num(value):
                try:
                    return f"{float(value):.2f}"
                except Exception:
                    return str(value)

            def _fmt_change(change, pct):
                try:
                    change_f = float(change)
                    pct_f = float(pct)
                    sign = "+" if change_f >= 0 else ""
                    return f"{sign}{change_f:.2f} ({sign}{pct_f:.2f}%)"
                except Exception:
                    return f"{change} ({pct}%)"

            if result.get("mode") == "symbol":
                quote = result.get("quote", {}) or {}
                return (
                    f"Indian stock details for {quote.get('name', quote.get('symbol', 'N/A'))} [{quote.get('symbol', 'N/A')}]:\n"
                    f"- Price: {quote.get('currency', 'INR')} {_fmt_num(quote.get('price'))}\n"
                    f"- Change: {_fmt_change(quote.get('change'), quote.get('change_percent'))}\n"
                    f"- Day Range: {_fmt_num(quote.get('day_low'))} - {_fmt_num(quote.get('day_high'))}\n"
                    f"- Exchange: {quote.get('exchange', 'N/A')}"
                )

            indices = result.get("indices", []) or []
            if not indices:
                return "No Indian market index data returned."
            lines = ["Indian market overview:"]
            for row in indices:
                lines.append(
                    f"- {row.get('name', row.get('symbol'))}: {_fmt_num(row.get('price'))} | {_fmt_change(row.get('change'), row.get('change_percent'))}"
                )
            return "\n".join(lines)

        if tool_name == "weather_fetcher" and isinstance(result, dict):
            if not result.get("ok"):
                return f"Weather fetch failed: {result.get('error', 'unknown error')}"
            location = result.get("location", "Unknown location")
            region = result.get("region", "")
            country = result.get("country", "")
            where_parts = [location]
            if region:
                where_parts.append(region)
            if country:
                where_parts.append(country)
            where = ", ".join(where_parts)
            condition = result.get("condition", "Unknown")
            temp_c = result.get("temp_c", "N/A")
            feels_c = result.get("feels_like_c", "N/A")
            humidity = result.get("humidity_pct", "N/A")
            wind = result.get("wind_kmph", "N/A")
            return (
                f"Current weather for {where}:\n"
                f"- Condition: {condition}\n"
                f"- Temperature: {temp_c} C (feels like {feels_c} C)\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind: {wind} km/h"
            )

        if tool_name == "news_fetcher" and isinstance(result, dict):
            if not result.get("ok"):
                return f"News fetch failed: {result.get('error', 'unknown error')}"
            topic = result.get("topic", "top headlines")
            headlines = result.get("headlines", []) or []
            if not headlines:
                return f"No headlines found for '{topic}'."
            lines = [f"Latest news for {topic}:"]
            for idx, item in enumerate(headlines, 1):
                title = item.get("title", "Untitled")
                source = item.get("source", "")
                link = item.get("link", "")
                src_suffix = f" ({source})" if source else ""
                lines.append(f"{idx}. {title}{src_suffix}")
                if link:
                    lines.append(f"   {link}")
            return "\n".join(lines)

        if isinstance(result, dict):
            payload = dict(result)
            if "content" in payload and isinstance(payload["content"], str):
                if len(payload["content"]) > 1800:
                    payload["content"] = payload["content"][:1800] + "\n... [truncated]"
            return f"Tool `{tool_name}` result:\n{json.dumps(payload, indent=2)}"
        return f"Tool `{tool_name}` result: {result}"

    def _execute_tool_if_requested(self, user_message: str, mode_obj) -> Optional[str]:
        """
        Tool decision layer:
        execute explicit tool requests reactively with no persistent workers.
        """
        if not self._is_tool_request(user_message):
            return None

        text = user_message.strip()
        if text.lower().startswith("/tool "):
            text = text[6:].strip()

        tool_call = self._tool_router.detect_intent(text)
        if not tool_call.tool_name:
            return None

        # In non-tool modes, allow safe read-only tools without /tool prefix.
        safe_tools = {"stock_fetcher", "news_fetcher", "weather_fetcher", "indian_market_fetcher", "current_affairs_fetcher"}
        if (
            not mode_obj.use_tools
            and not user_message.strip().lower().startswith("/tool ")
            and tool_call.tool_name not in safe_tools
        ):
            return None

        try:
            result = self._tool_registry.execute_tool(
                tool_call.tool_name,
                dict(tool_call.parameters or {}),
                require_confirmation=False
            )
        except Exception as exc:
            return f"Tool `{tool_call.tool_name}` failed: {exc}"
        if result.get("status") != "ok":
            return f"Tool `{tool_call.tool_name}` failed: {result.get('error') or result.get('message')}"

        return self._format_tool_response(tool_call.tool_name, result.get("result"))

    def _direct_memory_recall_response(self, user_message: str, runtime: Dict) -> Optional[str]:
        """
        Deterministic recall path for direct personal-memory questions.
        """
        if not runtime.get("memory_enabled", True):
            return None

        text = (user_message or "").strip().lower()
        if not text:
            return None

        asks_name = bool(re.search(r"\b(what(?:'s| is)\s+my\s+name|do\s+you\s+remember\s+my\s+name)\b", text))
        words = text.split()
        is_question_like = bool(text.endswith("?")) or (words[0] in {"what", "which", "do", "can", "could", "tell", "remind"} if words else False)
        asks_difficulties = is_question_like and (
            "difficulties" in text
            or "struggle" in text
            or "weak" in text
            or ("subjects" in text and "difficult" in text)
        )

        if asks_name:
            name = self._memory_manager.get_structured_memory("user.name")
            if name:
                return f"Your name is {name}."
            return "I do not have your name saved yet."

        if asks_difficulties:
            difficulties = self._memory_manager.get_structured_memory("system_state.difficulties") or []
            if isinstance(difficulties, list) and difficulties:
                return "You previously mentioned difficulties in: " + ", ".join(str(x) for x in difficulties) + "."
            return "I do not have any saved difficulty subjects yet."

        return None

    def _direct_datetime_response(self, user_message: str) -> Optional[str]:
        """
        Deterministic realtime date/time response for clock-style queries.
        """
        text = (user_message or "").strip().lower()
        if not text:
            return None

        asks_time = bool(
            re.search(
                r"\b(what(?:'s| is)?\s+the\s+time|what\s+time\s+is\s+it|what(?:'s| is)?\s+time\s+now|current\s+time|time\s+now)\b",
                text,
            )
        )
        asks_date = bool(
            re.search(
                r"\b(what(?:'s| is)?\s+the\s+date|what\s+date\s+is\s+it|today(?:'s)?\s+date|current\s+date|what\s+day\s+is\s+it)\b",
                text,
            )
        )

        if not (asks_time or asks_date):
            return None

        now_local = datetime.now().astimezone()
        day_str = now_local.strftime("%A, %B %d, %Y")
        time_str = now_local.strftime("%I:%M %p")
        tz = now_local.tzname() or "local time"

        if asks_time and asks_date:
            return f"Current local date and time: {day_str}, {time_str} ({tz})."
        if asks_time:
            return f"Current local time: {time_str} ({tz}) on {day_str}."
        return f"Current local date: {day_str}. Local time is {time_str} ({tz})."
    
    def generate(self, user_message: str, mode: str = "chat", options: Optional[Dict] = None) -> str:
        """
        Generate a response for the user message.
        
        Args:
            user_message: The user's input message
            mode: Operation mode (chat, coding, research, agent)
            
        Returns:
            Generated response string
        """
        resolved_mode, mode_obj = self._resolve_mode(user_message, mode)
        runtime = self._resolve_runtime_options(options)
        slash_result = self._execute_slash_command(user_message, mode_obj, runtime)
        if slash_result is not None:
            response_text = slash_result.get("response_text", "")
            command = slash_result["command"]
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"] and command.store_in_memory:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', response_text)
                self._memory_manager.save_all()
            return response_text

        datetime_reply = self._direct_datetime_response(user_message)
        if datetime_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', datetime_reply)
                self._memory_manager.save_all()
            return datetime_reply

        tool_reply = self._execute_tool_if_requested(user_message, mode_obj)
        if tool_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', tool_reply)
                self._memory_manager.save_all()
            return tool_reply

        memory_reply = self._direct_memory_recall_response(user_message, runtime)
        if memory_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', memory_reply)
                self._memory_manager.save_all()
            return memory_reply

        prep = self._prepare_generation_context(user_message=user_message, mode=resolved_mode, options=runtime, stream=False)
        mode = prep['mode']
        mode_obj = prep['mode_obj']
        prompt = prep['prompt']
        runtime = prep['runtime']
        memory_updated = bool(prep.get('memory_updated', False))
        self._last_turn_meta = {"memory_updated": memory_updated}
        
        # Generate response via adapter
        start = time.time()
        try:
            result = self._model_adapter.generate(
                prompt, 
                max_tokens=mode_obj.max_tokens, 
                temperature=mode_obj.temperature
            )
            reply = result.text.strip()
        except Exception as e:
            import traceback
            duration = time.time() - start
            
            logger.error(f"Model generation failed after {duration:.2f}s:")
            traceback.print_exc()
            
            if DEBUG_MODE:
                reply = f"ERROR: {type(e).__name__}: {str(e)}"
            else:
                reply = "Sorry, I couldn't generate a response at this time. Please try again later."
            
            # Keep short-term record for failed turn
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('assistant', reply)
                self._memory_manager.save_all()
            return reply
        
        duration = time.time() - start
        
        # Log generation metrics
        try:
            tokens_out = len(reply.split())
        except Exception:
            tokens_out = 0
        
        tps = tokens_out / duration if duration > 0 else tokens_out
        logger.info(f"Mode={mode} generated {tokens_out} tokens in {duration:.2f}s ({tps:.1f} tokens/sec)")
        
        # ================================================
        # UPDATE MEMORY AFTER RESPONSE
        # ================================================
        if runtime["memory_enabled"]:
            self._memory_manager.add_short_term_message('assistant', reply)
            self._memory_manager.save_all()
        
        # ================================================
        # AGENT MODE: Tool Execution
        # ================================================
        if mode_obj.name == 'agent' and mode_obj.use_tools:
            reply = self._handle_agent_mode(reply)
        
        return reply
    
    def run_stream(self, user_message: str, mode: str = "chat", options: Optional[Dict] = None) -> Generator[Dict]:
        """
        Streaming generation - yields chunks as they arrive.
        
        Reuses context preparation from generate() via private methods.
        Yields dict chunks with 'content' and optionally 'done' + token counts.
        
        Args:
            user_message: The user's input message
            mode: Operation mode
            
        Yields:
            Dict with 'content' key for streaming chunks
            Final chunk includes 'done', 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        resolved_mode, mode_obj = self._resolve_mode(user_message, mode)
        runtime = self._resolve_runtime_options(options)
        slash_result = self._execute_slash_command(user_message, mode_obj, runtime)
        if slash_result is not None:
            response_text = slash_result.get("response_text", "")
            command = slash_result["command"]
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"] and command.store_in_memory:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', response_text)
                self._memory_manager.save_all()
            prompt_tokens = self._model_adapter.estimate_tokens(user_message)
            completion_tokens = self._model_adapter.estimate_tokens(response_text)
            yield {'content': response_text}
            yield {
                'done': True,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'memory_updated': False,
            }
            return

        datetime_reply = self._direct_datetime_response(user_message)
        if datetime_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', datetime_reply)
                self._memory_manager.save_all()
            completion_tokens = self._model_adapter.estimate_tokens(datetime_reply)
            yield {'content': datetime_reply}
            yield {
                'done': True,
                'prompt_tokens': 0,
                'completion_tokens': completion_tokens,
                'total_tokens': completion_tokens,
                'memory_updated': False,
            }
            return

        tool_reply = self._execute_tool_if_requested(user_message, mode_obj)
        if tool_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', tool_reply)
                self._memory_manager.save_all()
            prompt_tokens = self._model_adapter.estimate_tokens(user_message)
            completion_tokens = self._model_adapter.estimate_tokens(tool_reply)
            yield {'content': tool_reply}
            yield {
                'done': True,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'memory_updated': False,
            }
            return

        memory_reply = self._direct_memory_recall_response(user_message, runtime)
        if memory_reply is not None:
            self._last_turn_meta = {"memory_updated": False}
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('user', user_message)
                self._memory_manager.add_short_term_message('assistant', memory_reply)
                self._memory_manager.save_all()
            completion_tokens = self._model_adapter.estimate_tokens(memory_reply)
            yield {'content': memory_reply}
            yield {
                'done': True,
                'prompt_tokens': 0,
                'completion_tokens': completion_tokens,
                'total_tokens': completion_tokens,
                'memory_updated': False,
            }
            return

        prep = self._prepare_generation_context(user_message=user_message, mode=resolved_mode, options=runtime, stream=True)
        mode = prep['mode']
        mode_obj = prep['mode_obj']
        prompt = prep['prompt']
        runtime = prep['runtime']
        memory_updated = bool(prep.get('memory_updated', False))
        self._last_turn_meta = {"memory_updated": memory_updated}
        
        # Estimate prompt tokens via adapter
        prompt_tokens = self._model_adapter.estimate_tokens(prompt)
        
        # Stream response via adapter
        start = time.time()
        full_response = ""
        stream_completed = False
        
        try:
            for chunk in self._model_adapter.generate_stream(
                prompt, 
                max_tokens=mode_obj.max_tokens, 
                temperature=mode_obj.temperature
            ):
                full_response += chunk
                yield {'content': chunk}
            
            reply = full_response.strip()
            stream_completed = True
            
        except Exception as e:
            import traceback
            duration = time.time() - start
            
            logger.error(f"Model streaming failed after {duration:.2f}s:")
            traceback.print_exc()
            
            if DEBUG_MODE:
                reply = f"ERROR: {type(e).__name__}: {str(e)}"
            else:
                reply = "Sorry, I couldn't generate a response at this time. Please try again later."
            
            yield {'content': reply}
            # Do NOT save memory for failed/partial streams
            # stream_completed remains False
        
        duration = time.time() - start
        
        # Calculate completion tokens via adapter
        completion_tokens = self._model_adapter.estimate_tokens(reply)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log generation metrics
        try:
            tokens_out = len(reply.split())
        except Exception:
            tokens_out = 0
        
        tps = tokens_out / duration if duration > 0 else tokens_out
        logger.info(f"Mode={mode} streamed {tokens_out} tokens in {duration:.2f}s ({tps:.1f} tokens/sec)")
        
        # ================================================
        # PHASE 5: MEMORY SAVED ONLY AFTER FULL STREAM
        # ================================================
        # Only save memory if stream completed successfully
        if stream_completed:
            if runtime["memory_enabled"]:
                self._memory_manager.add_short_term_message('assistant', full_response)
                self._memory_manager.save_all()
            
            logger.debug("Stream completed - memory saved")
        else:
            # Log that we skipped saving due to failure
            logger.warning("Stream did not complete - memory NOT saved to prevent partial data")
        
        # ================================================
        # AGENT MODE: Tool Execution
        # ================================================
        if mode_obj.name == 'agent' and mode_obj.use_tools and stream_completed:
            reply = self._handle_agent_mode(reply)
        
        # Yield final chunk with token counts
        done_payload = {
            'done': True,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'memory_updated': memory_updated,
        }
        if runtime.get("dev_logs"):
            done_payload["runtime"] = {
                "mode": mode,
                "project": runtime.get("project"),
                "memory_enabled": runtime.get("memory_enabled"),
                "duration_ms": int(duration * 1000),
            }
        yield done_payload

    def get_last_turn_meta(self) -> Dict[str, bool]:
        """Return metadata from the most recent generation turn."""
        return dict(self._last_turn_meta)
    
    # ===================
    # Helper Methods
    # ===================
    
    def _is_personal_message(self, user_input: str) -> bool:
        """Check if message contains personal information."""
        personal_keywords = ["my name is", "i was born", "my birthday is"]
        return any(keyword in user_input.lower() for keyword in personal_keywords)
    
    def _extract_name(self) -> Optional[str]:
        """Extract user name from structured memory."""
        return self._memory_manager.get_structured_memory('user.name')
    
    def _extract_birth_year(self) -> Optional[int]:
        """Extract birth year from structured memory."""
        return self._memory_manager.get_structured_memory('user.birth_year')
    
    def _extract_target_year(self, text: str) -> Optional[int]:
        """Extract future year from text."""
        for word in text.split():
            if word.isdigit() and len(word) == 4:
                return int(word)
        return None
    
    def _calculate_age(self, birth_year: Optional[int], year: int = None) -> Optional[int]:
        """Calculate age from birth year."""
        if not birth_year:
            return None
        year = year or datetime.now().year
        return year - birth_year
    
    def _is_study_query(self, user_input: str) -> bool:
        """Check if input is study-related."""
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in self._study_keywords)
    
    def _should_retrieve_documents(self, user_input: str) -> bool:
        """Determine whether to run document retrieval."""
        if not user_input:
            return False
        
        text = user_input.strip()
        
        # Question mark
        if text.endswith('?'):
            return True
        
        # Question words
        question_starts = ('what', 'why', 'how', 'when', 'where', 'who', 'do', 'does', 'did', 'can', 'could')
        if text.lower().split()[0] in question_starts:
            return True

        # Study retrieval only when explicitly asking to explain/solve/derive/etc.
        lowered = text.lower()
        retrieval_verbs = ("explain", "solve", "derive", "calculate", "formula", "difference")
        if any(v in lowered for v in retrieval_verbs) and self._is_study_query(user_input):
            return True

        return False
    
    def _build_system_message(
        self, 
        mode_obj, 
        name: Optional[str], 
        birth_year: Optional[int], 
        age: Optional[int], 
        current_year: int, 
        future_age: Optional[int],
        project: str,
        now_local: datetime
    ) -> str:
        """Build system message with user facts."""
        personality = self._config.personality_persistent_prompt
        mode_prompt = mode_obj.system_prompt or ""
        date_str = now_local.strftime("%A, %B %d, %Y")
        time_str = now_local.strftime("%I:%M %p")
        tz = now_local.tzname() or "local time"
        return f"""{personality}

Mode Instructions:
{mode_prompt}

Memory Instructions:
- Treat structured memory as source-of-truth for personal facts and prior stated difficulties.
- When asked about remembered user details, answer directly from memory context if available.

System Facts:
- Current Year: {current_year}
- Current Date: {date_str}
- Current Local Time: {time_str} ({tz})
- Active Project: {project}
- Name: {name}
- Birth Year: {birth_year}
- Current Age: {age}
- Future Age (if applicable): {future_age}

Use system facts when answering.
Be clear and natural."""
    
    def _handle_agent_mode(self, reply: str) -> str:
        """Handle agent mode tool execution."""
        try:
            parsed = json.loads(reply)
            if isinstance(parsed, dict) and parsed.get('action'):
                action = parsed.get('action')
                params = parsed.get('parameters', {})
                
                try:
                    exec_result = self._tool_registry.execute_tool(
                        action, 
                        params, 
                        require_confirmation=True
                    )
                    
                    if exec_result.get('status') == 'requires_confirmation':
                        return f"Proposed action requires confirmation: {json.dumps(parsed)}"
                    elif exec_result.get('status') == 'ok':
                        return f"Action executed: {action} -> {exec_result.get('result')}"
                    else:
                        return f"Tool execution error: {exec_result.get('error')}"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Tool execution failed for action '{action}': {e}")
                    
                    if DEBUG_MODE:
                        return f"Tool execution failed: {type(e).__name__}: {str(e)}"
                    else:
                        return f"Tool execution error: {action}"
        except json.JSONDecodeError:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Agent mode error: {e}")
        
        return reply
