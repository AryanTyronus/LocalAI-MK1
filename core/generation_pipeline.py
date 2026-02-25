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
from datetime import datetime
from typing import Generator, Dict, List, Optional

from core.config import Config, DEBUG_MODE
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
        
        # Phase 2: TokenBudgetManager for context enforcement
        self._token_budget_manager = token_budget_manager or TokenBudgetManager(config)
        
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
        
        logger.info("GenerationPipeline initialized")
    
    # ===================
    # Main Generation Methods
    # ===================
    
    def generate(self, user_message: str, mode: str = "chat") -> str:
        """
        Generate a response for the user message.
        
        Args:
            user_message: The user's input message
            mode: Operation mode (chat, coding, research, agent)
            
        Returns:
            Generated response string
        """
        # Auto-route intent if mode is 'auto' or None
        if not mode or mode == 'auto':
            inferred, conf = IntentClassifier.classify(user_message)
            logger.info(f"Auto-routed intent: {inferred} (conf={conf:.2f})")
            mode = inferred
        
        # Get mode configuration
        mode_obj = self._mode_controller.get_mode(mode)
        
        # Clear short-term memory for personal messages
        if self._is_personal_message(user_message):
            self._memory_manager.clear_short_term()
        
        # ================================================
        # LAYER 1: SHORT-TERM MEMORY
        # ================================================
        self._memory_manager.add_short_term_message('user', user_message)
        
        # Auto-extract profile information
        self._memory_manager.extract_profile_facts(user_message)
        
        # IMMEDIATELY save extracted facts to disk - don't wait
        self._memory_manager.save_all()
        
        # Check if we should create rolling summary
        created_summary = self._memory_manager.maybe_create_summary()
        if created_summary:
            logger.info("Created rolling summary from short-term memory")
        
        # ================================================
        # SEMANTIC MEMORY & FACTS EXTRACTION
        # ================================================
        user_lower = user_message.lower()
        if any(trigger in user_lower for trigger in self._memory_triggers):
            self._memory_manager.add_semantic_memory(user_message)
            # IMMEDIATELY save semantic memory to disk - don't wait for response
            self._memory_manager.save_all()
        
        # ================================================
        # GATHER CONTEXT FROM ALL LAYERS
        # ================================================
        name = self._extract_name()
        birth_year = self._extract_birth_year()
        current_year = datetime.now().year
        age = self._calculate_age(birth_year)
        
        # Detect future year queries
        target_year = self._extract_target_year(user_lower)
        
        future_age = None
        if birth_year and target_year:
            future_age = self._calculate_age(birth_year, target_year)
        
        # Get semantic memory context
        try:
            memory_context = self._memory_manager.get_semantic_context(user_message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Memory retrieval failed: {e}")
            memory_context = ""
        
        # Retrieve relevant document chunks when needed
        docs = []
        if self._should_retrieve_documents(user_message):
            try:
                active_doc = self._document_manager.active_doc_id if self._document_manager else None
                search_results = []
                if self._document_manager:
                    search_results = self._document_manager.search(
                    user_message, 
                    top_k=5, 
                    doc_id=active_doc
                )
                docs = search_results
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Document retrieval failed: {e}")
                docs = []
        
        # ================================================
        # BUILD PROMPT WITH ALL CONTEXT
        # ================================================
        full_context = self._memory_manager.build_full_context(user_message)
        
        # Build retrieved document context block
        retrieved_block = ""
        if docs:
            parts = ["=== Retrieved Document Context ==="]
            for res in docs:
                chunk = res.get('chunk', {})
                meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                parts.append(f"{meta}\n{chunk.get('text')}")
            retrieved_block = "\n\n".join(parts)
        
        # Determine which memory layers to include based on memory_profile
        include_structured = True
        include_rolling = True
        include_semantic = True
        
        memory_profile = mode_obj.memory_profile
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
        
        # Assemble context parts
        context_parts = []
        if include_structured:
            context_parts.append(full_context.get('structured', ''))
        if retrieved_block:
            context_parts.append(retrieved_block)
        if include_rolling:
            context_parts.append(full_context.get('rolling_summary', ''))
        if include_semantic:
            context_parts.append(full_context.get('semantic', ''))
        context_parts.append(f"Current Conversation:\n{full_context.get('short_term', '')}")
        
        context_str = "\n\n".join(p for p in context_parts if p)
        
        # Build system message
        system_message = self._build_system_message(
            mode_obj, name, birth_year, age, current_year, future_age
        )
        
        # ================================================
        # PHASE 2: TOKEN BUDGET ENFORCEMENT
        # ================================================
        # Extract context components for budget calculation
        structured_mem = full_context.get('structured', '') if include_structured else ''
        rolling_summary_mem = full_context.get('rolling_summary', '') if include_rolling else ''
        semantic_mem = full_context.get('semantic', '') if include_semantic else ''
        short_term_conv = full_context.get('short_term', '')
        
        # Calculate budget and get trimmed context if needed
        trimmed_context, budget, trimming_log = self._token_budget_manager.get_trimmed_context(
            system_prompt=mode_obj.system_prompt or system_message,
            structured_memory=structured_mem,
            rolling_summary=rolling_summary_mem,
            semantic_memories=semantic_mem,
            documents=retrieved_block,
            short_term=short_term_conv,
            user_query=user_message
        )
        
        # Log trimming events
        if trimming_log.get('trimmed_blocks'):
            logger.info(f"Token budget: {budget.total} tokens, trimming {len(trimming_log['trimmed_blocks'])} blocks")
            for tb in trimming_log['trimmed_blocks']:
                logger.info(f"  - {tb['name']}: {tb['original_tokens']} -> {tb['remaining_tokens']} tokens")
        
        # ================================================
        # PHASE 6: Build prompt using PromptBuilder with token-budget-aware context
        # ================================================
        built = self._prompt_builder.build(
            system_prompt=trimmed_context.get('system_prompt', mode_obj.system_prompt or system_message),
            retrieved_docs=docs,
            memories={
                'structured': trimmed_context.get('structured', ''),
                'rolling_summary': trimmed_context.get('rolling_summary', ''),
                'semantic': trimmed_context.get('semantic', '')
            },
            recent_conv=trimmed_context.get('short_term', ''),
            user_message=user_message,
        )
        
        # Log full context if debug enabled
        if DEBUG_MODE:
            logger.debug(f"[DEBUG] Full prompt context:\n{built['system']}\n\n{built['user']}")
        
        # Format prompt via adapter
        prompt = self._model_adapter.format_chat(
            system=built['system'],
            user=built['user'],
            history=None
        )
        
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
            
            # Still add to memory even on error
            self._memory_manager.add_user_message(user_message)
            self._memory_manager.add_assistant_message(reply)
            self._memory_manager.persist_structured_memory()
            self._memory_manager.persist_semantic_memory()
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
        self._memory_manager.add_user_message(user_message)
        self._memory_manager.add_assistant_message(reply)
        
        self._memory_manager.persist_structured_memory()
        self._memory_manager.persist_semantic_memory()
        
        # ================================================
        # AGENT MODE: Tool Execution
        # ================================================
        if mode_obj.name == 'agent' and mode_obj.use_tools:
            reply = self._handle_agent_mode(reply)
        
        return reply
    
    def run_stream(self, user_message: str, mode: str = "chat") -> Generator[Dict]:
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
        # Auto-route intent if mode is 'auto' or None
        if not mode or mode == 'auto':
            inferred, conf = IntentClassifier.classify(user_message)
            logger.info(f"Auto-routed intent: {inferred} (conf={conf:.2f})")
            mode = inferred
        
        # Get mode configuration
        mode_obj = self._mode_controller.get_mode(mode)
        
        # Clear short-term memory for personal messages
        if self._is_personal_message(user_message):
            self._memory_manager.clear_short_term()
        
        # ================================================
        # LAYER 1: SHORT-TERM MEMORY
        # ================================================
        self._memory_manager.add_short_term_message('user', user_message)
        
        # Auto-extract profile information
        self._memory_manager.extract_profile_facts(user_message)
        
        # IMMEDIATELY save extracted facts to disk - don't wait
        self._memory_manager.save_all()
        
        # Check if we should create rolling summary
        created_summary = self._memory_manager.maybe_create_summary()
        if created_summary:
            logger.info("Created rolling summary from short-term memory")
        
        # ================================================
        # SEMANTIC MEMORY & FACTS EXTRACTION
        # ================================================
        user_lower = user_message.lower()
        if any(trigger in user_lower for trigger in self._memory_triggers):
            self._memory_manager.add_semantic_memory(user_message)
            # IMMEDIATELY save semantic memory to disk - don't wait for stream completion
            self._memory_manager.save_all()
        
        # ================================================
        # GATHER CONTEXT FROM ALL LAYERS
        # ================================================
        name = self._extract_name()
        birth_year = self._extract_birth_year()
        current_year = datetime.now().year
        age = self._calculate_age(birth_year)
        
        # Detect future year queries
        target_year = self._extract_target_year(user_lower)
        
        future_age = None
        if birth_year and target_year:
            future_age = self._calculate_age(birth_year, target_year)
        
        # Get semantic memory context
        try:
            memory_context = self._memory_manager.get_semantic_context(user_message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Memory retrieval failed: {e}")
            memory_context = ""
        
        # Retrieve relevant document chunks when needed
        docs = []
        if self._should_retrieve_documents(user_message):
            try:
                active_doc = self._document_manager.active_doc_id if self._document_manager else None
                search_results = []
                if self._document_manager:
                    search_results = self._document_manager.search(
                    user_message, 
                    top_k=5, 
                    doc_id=active_doc
                )
                docs = search_results
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Document retrieval failed: {e}")
                docs = []
        
        # ================================================
        # BUILD PROMPT WITH ALL CONTEXT
        # ================================================
        full_context = self._memory_manager.build_full_context(user_message)
        
        # Build retrieved document context block
        retrieved_block = ""
        if docs:
            parts = ["=== Retrieved Document Context ==="]
            for res in docs:
                chunk = res.get('chunk', {})
                meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                parts.append(f"{meta}\n{chunk.get('text')}")
            retrieved_block = "\n\n".join(parts)
        
        # Determine which memory layers to include based on memory_profile
        include_structured = True
        include_rolling = True
        include_semantic = True
        
        memory_profile = mode_obj.memory_profile
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
        
        # Assemble context parts
        context_parts = []
        if include_structured:
            context_parts.append(full_context.get('structured', ''))
        if retrieved_block:
            context_parts.append(retrieved_block)
        if include_rolling:
            context_parts.append(full_context.get('rolling_summary', ''))
        if include_semantic:
            context_parts.append(full_context.get('semantic', ''))
        context_parts.append(f"Current Conversation:\n{full_context.get('short_term', '')}")
        
        context_str = "\n\n".join(p for p in context_parts if p)
        
        # Build system message
        system_message = self._build_system_message(
            mode_obj, name, birth_year, age, current_year, future_age
        )
        
        # ================================================
        # PHASE 2: TOKEN BUDGET ENFORCEMENT (same as generate)
        # ================================================
        structured_mem = full_context.get('structured', '') if include_structured else ''
        rolling_summary_mem = full_context.get('rolling_summary', '') if include_rolling else ''
        semantic_mem = full_context.get('semantic', '') if include_semantic else ''
        short_term_conv = full_context.get('short_term', '')
        
        trimmed_context, budget, trimming_log = self._token_budget_manager.get_trimmed_context(
            system_prompt=mode_obj.system_prompt or system_message,
            structured_memory=structured_mem,
            rolling_summary=rolling_summary_mem,
            semantic_memories=semantic_mem,
            documents=retrieved_block,
            short_term=short_term_conv,
            user_query=user_message
        )
        
        # Log trimming events
        if trimming_log.get('trimmed_blocks'):
            logger.info(f"Token budget (stream): {budget.total} tokens, trimming {len(trimming_log['trimmed_blocks'])} blocks")
        
        # ================================================
        # PHASE 6: Build prompt with token-budget-aware context
        # ================================================
        built = self._prompt_builder.build(
            system_prompt=trimmed_context.get('system_prompt', mode_obj.system_prompt or system_message),
            retrieved_docs=docs,
            memories={
                'structured': trimmed_context.get('structured', ''),
                'rolling_summary': trimmed_context.get('rolling_summary', ''),
                'semantic': trimmed_context.get('semantic', '')
            },
            recent_conv=trimmed_context.get('short_term', ''),
            user_message=user_message,
        )
        
        # Log full context if debug enabled
        if DEBUG_MODE:
            logger.debug(f"[DEBUG] Full prompt context (stream):\n{built['system']}\n\n{built['user']}")
        
        # Format prompt via adapter
        prompt = self._model_adapter.format_chat(
            system=built['system'],
            user=built['user'],
            history=None
        )
        
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
            
            # IMMEDIATELY save assistant message right after stream completes
            self._memory_manager.add_short_term_message('assistant', full_response)
            self._memory_manager.save_all()
            
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
            self._memory_manager.add_user_message(user_message)
            self._memory_manager.add_assistant_message(full_response)
            
            self._memory_manager.persist_structured_memory()
            self._memory_manager.persist_semantic_memory()
            
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
        yield {
            'done': True,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }
    
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
        
        # Study queries
        return self._is_study_query(user_input)
    
    def _build_system_message(
        self, 
        mode_obj, 
        name: Optional[str], 
        birth_year: Optional[int], 
        age: Optional[int], 
        current_year: int, 
        future_age: Optional[int]
    ) -> str:
        """Build system message with user facts."""
        return f"""System Facts:
- Current Year: {current_year}
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

