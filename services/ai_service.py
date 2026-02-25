from datetime import datetime
from core.config import MAX_HISTORY_TURNS, STUDY_KEYWORDS, get_mode_config
from core.chat_formatter import Message, Role
from memory.memory_manager import MemoryManager
from memory.memory_store import MemoryStore
from retrieval.document_manager import DocumentManager
from core.model_manager import ModelManager
from core.logger import logger

# New modular components
from core.mode_controller import ModeController
from core.prompt_builder import PromptBuilder
from core.tool_registry import ToolRegistry
from core.intent_classifier import IntentClassifier
import time
import json


class AIService:
    """
    AI Service with integrated 4-layer memory architecture.
    Maintains backward compatibility while supporting new memory features.
    """

    def __init__(
        self,
        config=None,
        memory_service=None,
        summarizer=None,
        model_manager=None,
        use_4layer_memory=True
    ):
        # MK-2 injection-ready (optional)
        self.config = config
        self.memory_service = memory_service
        self.summarizer = summarizer

        # Preserve original singleton behavior
        self.model_manager = model_manager or ModelManager.get_instance()

        # Initialize 4-layer memory system
        self.use_4layer_memory = use_4layer_memory
        if use_4layer_memory:
            self.memory_manager = MemoryManager()
            logger.info("Using 4-layer memory architecture")
        else:
            # Fallback to legacy memory system
            self.memory_store = memory_service.store if memory_service else MemoryStore()
            logger.info("Using legacy memory system")

        # Preserve original systems
        # DocumentManager handles PDF store, chunking and retrieval
        self.doc_manager = DocumentManager()

        self.chat_history = []  # Legacy - kept for compatibility

    # ---- PUBLIC ENTRY ----
    def ask(self, user_input, mode: str = "chat"):
        """Public entrypoint for asking a question with an optional mode.

        mode: one of 'chat', 'coding', 'research' (defaults to 'chat')
        """
        return self._generate(user_input, mode=mode)

    # ---- GENERATION LOGIC WITH 4-LAYER MEMORY ----
    def _generate(self, user_input, mode: str = "chat"):

        user_lower = user_input.lower()

        # Auto-route intent when mode is 'auto' or None
        if not mode or mode == 'auto':
            inferred, conf = IntentClassifier.classify(user_input)
            logger.info(f"Auto-routed intent: {inferred} (conf={conf:.2f})")
            mode = inferred

        # Get mode object with per-mode settings
        mode_obj = ModeController.get_mode(mode)
        if self._is_personal_message(user_input):
            if self.use_4layer_memory:
                self.memory_manager.clear_short_term()
            else:
                self.chat_history.clear()

        # ================================================
        # LAYER 1: SHORT-TERM MEMORY
        # ================================================
        if self.use_4layer_memory:
            self.memory_manager.add_short_term_message('user', user_input)

            # Auto-extract profile information
            self.memory_manager.extract_profile_facts(user_input)

            # Check if we should create rolling summary
            created_summary = self.memory_manager.maybe_create_summary()
            if created_summary:
                logger.info("Created rolling summary from short-term memory")
        else:
            self.chat_history.append(f"User: {user_input}")

        # ================================================
        # SEMANTIC MEMORY & FACTS EXTRACTION
        # ================================================
        memory_triggers = [
            "my name is",
            "i was born",
            "my birthday is",
            "i like",
            "i struggle"
        ]

        if self.use_4layer_memory:
            # Add to semantic memory
            for trigger in memory_triggers:
                if trigger in user_lower:
                    self.memory_manager.add_semantic_memory(user_input)
                    break
        else:
            if any(trigger in user_lower for trigger in memory_triggers):
                self.memory_store.add_memory(user_input)

        # ================================================
        # GATHER CONTEXT FROM ALL LAYERS
        # ================================================
        name = self._extract_name()
        birth_year = self._extract_birth_year()
        current_year = datetime.now().year
        age = self._calculate_age(birth_year)

        # Detect future year queries
        target_year = None
        for word in user_lower.split():
            if word.isdigit() and len(word) == 4:
                target_year = int(word)

        future_age = None
        if birth_year and target_year:
            future_age = self._calculate_age(birth_year, target_year)

        # Get memory from appropriate layer
        if self.use_4layer_memory:
            try:
                memory_context = self.memory_manager.get_semantic_context(user_input)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Memory retrieval failed: {e}")
                memory_context = ""
        else:
            try:
                memory_context = self.memory_store.search(user_input)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Legacy memory search failed: {e}")
                memory_context = ""

        # Retrieve relevant document chunks when the user asks a question
        docs = []
        if self._should_retrieve_documents(user_input):
            try:
                # If active document selected, search only that document
                active_doc = self.doc_manager.active_doc_id
                search_results = self.doc_manager.search(user_input, top_k=5, doc_id=active_doc)
                # search_results is list of {'chunk': {...}, 'score': float}
                docs = search_results
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Document retrieval failed: {e}")
                docs = []  # Continue without document context

        # ================================================
        # BUILD PROMPT WITH ALL CONTEXT
        # ================================================
        if self.use_4layer_memory:
            # Use full multi-layer context
            full_context = self.memory_manager.build_full_context(user_input)

            # Build retrieved document context block (do NOT store document text in chat memory)
            retrieved_block = ""
            if docs:
                parts = ["=== Retrieved Document Context ==="]
                for res in docs:
                    chunk = res.get('chunk', {})
                    meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                    # include only the chunk text in the prompt (not stored in memory)
                    parts.append(f"{meta}\n{chunk.get('text')}")
                retrieved_block = "\n\n".join(parts)

            context_parts = [
                full_context['structured'],
                retrieved_block,
                full_context['rolling_summary'],
                full_context['semantic'],
                f"Current Conversation:\n{full_context['short_term']}"
            ]
            context_str = "\n\n".join(p for p in context_parts if p)
        else:
            # Legacy approach
            self.chat_history.append(f"User: {user_input}")
            history_text = "\n".join(
                self.chat_history[-MAX_HISTORY_TURNS * 2:]
            )
            context_str = f"Recent Conversation:\n{history_text}\n\nMemories:\n{memory_context}"

        # Mode-specific configuration via ModeController
        mode_system = mode_obj.system_prompt
        temperature = float(mode_obj.temperature)
        retrieve_documents = bool(mode_obj.retrieve_documents)
        memory_profile = mode_obj.memory_profile

        # Build system message (mode-specific override or default)
        if mode_system:
            system_message = f"""{mode_system}\n\nSystem Facts:\n- Current Year: {current_year}\n- Name: {name}\n- Birth Year: {birth_year}\n- Current Age: {age}\n- Future Age (if applicable): {future_age}\n\nUse system facts when answering.\nBe clear and natural."""
        else:
            system_message = f"""You are a helpful personal AI assistant.\n\nSystem Facts:\n- Current Year: {current_year}\n- Name: {name}\n- Birth Year: {birth_year}\n- Current Age: {age}\n- Future Age (if applicable): {future_age}\n\nUse system facts when answering.\nBe clear and natural."""

        # Use model formatter to build proper prompt
        # Determine which memory layers to include based on memory_profile
        include_structured = True
        include_rolling = True
        include_semantic = True
        if memory_profile == 'short_term_minimal':
            include_structured = False
            include_rolling = False
            include_semantic = False
        elif memory_profile == 'short_term_heavy':
            include_structured = True
            include_rolling = False
            include_semantic = False
        elif memory_profile == 'rolling_summary_heavy':
            include_structured = True
            include_rolling = True
            include_semantic = True

        # Build retrieved document block only if retrieval is enabled by mode and earlier heuristic
        retrieved_block = ""
        if retrieve_documents and docs:
            parts = ["=== Retrieved Document Context ==="]
            for res in docs:
                chunk = res.get('chunk', {})
                meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                parts.append(f"{meta}\n{chunk.get('text')}")
            retrieved_block = "\n\n".join(parts)

        # Assemble context parts according to selected memory layers
        context_parts = []
        if include_structured:
            context_parts.append(full_context['structured'])
        if retrieved_block:
            context_parts.append(retrieved_block)
        if include_rolling:
            context_parts.append(full_context['rolling_summary'])
        if include_semantic:
            context_parts.append(full_context['semantic'])
        context_parts.append(f"Current Conversation:\n{full_context['short_term']}")

        context_str = "\n\n".join(p for p in context_parts if p)

        # Build prompt using PromptBuilder
        built = PromptBuilder.build(
            system_prompt=mode_system or system_message,
            retrieved_docs=docs,
            memories={
                'structured': full_context.get('structured', ''),
                'rolling_summary': full_context.get('rolling_summary', ''),
                'semantic': full_context.get('semantic', '')
            },
            recent_conv=full_context.get('short_term', ''),
            user_message=user_input,
        )

        prompt = self.model_manager.format_chat(
            system=built['system'],
            user=built['user'],
            history=None
        )

        # Generate and measure, with safe error handling
        start = time.time()
        try:
            response = self.model_manager.generate(prompt, max_tokens=mode_obj.max_tokens, temperature=temperature)
            reply = response.strip()
        except Exception as e:
            import traceback
            duration = time.time() - start
            
            # Always print full stack trace to console
            logger.error(f"Model generation failed after {duration:.2f}s:")
            traceback.print_exc()
            print(f"\n=== GENERATION ERROR DETAILS ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Duration: {duration:.2f}s")
            print(f"Mode: {mode}")
            print(f"=====================================\n")
            
            # Import DEBUG config
            from core.config import DEBUG_MODE
            
            # In DEBUG mode, return error details; in production, use fallback message
            if DEBUG_MODE:
                reply = f"ERROR: {type(e).__name__}: {str(e)}"
            else:
                reply = "Sorry, I couldn't generate a response at this time. Please try again later."
            
            # Persist state
            if self.use_4layer_memory:
                self.memory_manager.add_short_term_message('assistant', reply)
                self.memory_manager.save_all()
            else:
                self.chat_history.append(f"Assistant: {reply}")
            return reply

        duration = time.time() - start

        # Approximate token counts for observability
        try:
            tokens_out = len(reply.split())
        except Exception:
            tokens_out = 0

        tps = tokens_out / duration if duration > 0 else tokens_out
        logger.info(f"Mode={mode} generated {tokens_out} tokens in {duration:.2f}s ({tps:.1f} tokens/sec)")
        logger.debug(f"Retrieved docs: {len(docs)}; memory_summary_count: {len(self.memory_manager.rolling_summaries) if self.use_4layer_memory else 0}")

        # ================================================
        # UPDATE MEMORY AFTER RESPONSE
        # ================================================
        if self.use_4layer_memory:
            self.memory_manager.add_short_term_message('assistant', reply)
            self.memory_manager.save_all()
        else:
            self.chat_history.append(f"Assistant: {reply}")

        # Agent mode: parse structured tool calls and execute if allowed
        if mode_obj.name == 'agent' and mode_obj.use_tools:
            # Expect a JSON action from the assistant
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, dict) and parsed.get('action'):
                    action = parsed.get('action')
                    params = parsed.get('parameters', {})
                    try:
                        # Execute via ToolRegistry (requires confirmation by default)
                        exec_result = ToolRegistry.execute_tool(action, params, require_confirmation=True)
                        if exec_result.get('status') == 'requires_confirmation':
                            # Ask user for confirmation; return the proposed action for UI
                            return f"Proposed action requires confirmation: {json.dumps(parsed)}"
                        elif exec_result.get('status') == 'ok':
                            return f"Action executed: {action} -> {exec_result.get('result')}"
                        else:
                            return f"Tool execution error: {exec_result.get('error')}"
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        logger.error(f"Tool execution failed for action '{action}': {e}")
                        
                        from core.config import DEBUG_MODE
                        if DEBUG_MODE:
                            return f"Tool execution failed: {type(e).__name__}: {str(e)}"
                        else:
                            return f"Tool execution error: {action}"
            except json.JSONDecodeError:
                # Not a JSON action; continue normally
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent mode error: {e}")
                
                from core.config import DEBUG_MODE
                if DEBUG_MODE:
                    raise

        return reply

    # ---- HELPER METHODS ----

    def _is_personal_message(self, user_input):
        personal_keywords = ["my name is", "i was born", "my birthday is"]
        return any(keyword in user_input.lower() for keyword in personal_keywords)

    def _extract_name(self):
        if self.use_4layer_memory:
            return self.memory_manager.get_structured_memory('user.name')
        else:
            for entry in self.chat_history:
                if "my name is" in entry.lower():
                    return entry.split("my name is")[-1].strip()
        return None

    def _extract_birth_year(self):
        if self.use_4layer_memory:
            return self.memory_manager.get_structured_memory('user.birth_year')
        else:
            for entry in self.chat_history:
                words = entry.split()
                for word in words:
                    if word.isdigit() and len(word) == 4:
                        return int(word)
        return None

    def _calculate_age(self, birth_year, year=None):
        if not birth_year:
            return None
        year = year or datetime.now().year
        return year - birth_year

    def _is_study_query(self, user_input):
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in STUDY_KEYWORDS)

    def _should_retrieve_documents(self, user_input: str) -> bool:
        """Determine whether to run document retrieval for this user input.
        We retrieve when the user asks a question or the input appears study-related.
        """
        if not user_input:
            return False

        # Simple heuristics: question mark or common question words
        text = user_input.strip()
        if text.endswith('?'):
            return True

        question_starts = ('what', 'why', 'how', 'when', 'where', 'who', 'do', 'does', 'did', 'can', 'could')
        if text.lower().split()[0] in question_starts:
            return True

        # Study queries also trigger document retrieval
        return self._is_study_query(user_input)
