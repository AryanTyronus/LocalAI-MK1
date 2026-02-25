"""
TokenBudgetManager - Token budgeting and context control system.

This module provides:
- Token estimation for all context blocks
- Enforcement of max context size before generation
- Dynamic trimming of lowest-priority context
- Context compression for rolling summaries and conversations
- Prevention of model truncation errors

Context blocks scored:
- System prompt
- Structured memory
- Rolling summaries
- Semantic memories
- Retrieved documents
- Short-term conversation
- User query
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.logger import logger
from core.config import Config


class ContextPriority(Enum):
    """Priority levels for context blocks."""
    CRITICAL = 1      # User query - never drop
    HIGH = 2          # System prompt, short-term conversation
    MEDIUM = 3        # Structured memory, retrieved documents
    LOW = 4           # Rolling summaries
    LOWEST = 5        # Semantic memories (can be trimmed heavily)


@dataclass
class ContextBlock:
    """A single context block with token count and priority."""
    name: str
    content: str
    tokens: int
    priority: ContextPriority
    priority_score: float = 0.0  # Computed score for sorting
    compressed: bool = False  # Whether block was compressed
    original_tokens: int = 0  # Original token count before compression


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    compression_type: str  # 'summary', 'conversation', 'semantic'


@dataclass
class TokenBudget:
    """Token budget breakdown."""
    system_prompt: int = 0
    structured: int = 0
    rolling_summary: int = 0
    semantic: int = 0
    documents: int = 0
    short_term: int = 0
    user_query: int = 0
    total: int = 0
    remaining: int = 0


class TokenBudgetManager:
    """
    Manages token budgets for context blocks.
    
    Responsibilities:
    - Estimate tokens for each context block
    - Enforce max context size
    - Dynamically trim lowest-priority context
    - Log all trimming events
    """
    
    # Default weights for token estimation
    # Based on average token length in English (~4 chars per token)
    CHARS_PER_TOKEN = 4.0
    
    # Priority order for trimming (first to last)
    TRIM_ORDER = [
        ContextPriority.LOWEST,   # Semantic memories
        ContextPriority.LOW,       # Rolling summaries  
        ContextPriority.MEDIUM,    # Documents
        ContextPriority.MEDIUM,    # Structured memory
        ContextPriority.HIGH,      # Short-term conversation
    ]
    
    def __init__(self, config: Config = None):
        """
        Initialize TokenBudgetManager.
        
        Args:
            config: Config instance for settings
        """
        self._config = config or Config()
        
        # Load settings from config
        self._max_context_tokens = self._config.max_context_tokens
        self._reserve_tokens = self._config.token_reserve_for_generation
        self._enable_trimming = self._config.token_trimming_enabled
        self._debug_mode = self._config.debug_enabled
        
        logger.info(f"TokenBudgetManager initialized: max={self._max_context_tokens}, reserve={self._reserve_tokens}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses character-based estimation for speed.
        Override this method for more accurate counting.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))
    
    def estimate_tokens_for_block(self, block: ContextBlock) -> int:
        """
        Estimate tokens for a context block.
        
        Args:
            block: ContextBlock to estimate
            
        Returns:
            Token count
        """
        return self.estimate_tokens(block.content)
    
    def build_context_blocks(
        self,
        system_prompt: str,
        structured_memory: str,
        rolling_summary: str,
        semantic_memories: str,
        documents: str,
        short_term: str,
        user_query: str
    ) -> List[ContextBlock]:
        """
        Build list of context blocks with priorities.
        
        Args:
            system_prompt: System prompt text
            structured_memory: Structured memory text
            rolling_summary: Rolling summary text
            semantic_memories: Semantic memories text
            documents: Retrieved documents text
            short_term: Short-term conversation
            user_query: User query
            
        Returns:
            List of ContextBlock objects
        """
        blocks = []
        
        # User query - CRITICAL (never trimmed)
        blocks.append(ContextBlock(
            name="user_query",
            content=user_query,
            tokens=self.estimate_tokens(user_query),
            priority=ContextPriority.CRITICAL,
            priority_score=1.0
        ))
        
        # System prompt - HIGH
        blocks.append(ContextBlock(
            name="system_prompt",
            content=system_prompt,
            tokens=self.estimate_tokens(system_prompt),
            priority=ContextPriority.HIGH,
            priority_score=0.9
        ))
        
        # Short-term conversation - HIGH
        blocks.append(ContextBlock(
            name="short_term",
            content=short_term,
            tokens=self.estimate_tokens(short_term),
            priority=ContextPriority.HIGH,
            priority_score=0.8
        ))
        
        # Structured memory - MEDIUM
        blocks.append(ContextBlock(
            name="structured",
            content=structured_memory,
            tokens=self.estimate_tokens(structured_memory),
            priority=ContextPriority.MEDIUM,
            priority_score=0.6
        ))
        
        # Retrieved documents - MEDIUM
        blocks.append(ContextBlock(
            name="documents",
            content=documents,
            tokens=self.estimate_tokens(documents),
            priority=ContextPriority.MEDIUM,
            priority_score=0.5
        ))
        
        # Rolling summaries - LOW
        blocks.append(ContextBlock(
            name="rolling_summary",
            content=rolling_summary,
            tokens=self.estimate_tokens(rolling_summary),
            priority=ContextPriority.LOW,
            priority_score=0.3
        ))
        
        # Semantic memories - LOWEST
        blocks.append(ContextBlock(
            name="semantic",
            content=semantic_memories,
            tokens=self.estimate_tokens(semantic_memories),
            priority=ContextPriority.LOWEST,
            priority_score=0.1
        ))
        
        return blocks
    
    def calculate_budget(
        self,
        system_prompt: str,
        structured_memory: str,
        rolling_summary: str,
        semantic_memories: str,
        documents: str,
        short_term: str,
        user_query: str
    ) -> TokenBudget:
        """
        Calculate token budget breakdown.
        
        Args:
            All context strings
            
        Returns:
            TokenBudget with breakdown
        """
        budget = TokenBudget(
            system_prompt=self.estimate_tokens(system_prompt),
            structured=self.estimate_tokens(structured_memory),
            rolling_summary=self.estimate_tokens(rolling_summary),
            semantic=self.estimate_tokens(semantic_memories),
            documents=self.estimate_tokens(documents),
            short_term=self.estimate_tokens(short_term),
            user_query=self.estimate_tokens(user_query)
        )
        
        budget.total = (
            budget.system_prompt +
            budget.structured +
            budget.rolling_summary +
            budget.semantic +
            budget.documents +
            budget.short_term +
            budget.user_query
        )
        
        budget.remaining = max(0, self._max_context_tokens - budget.total)
        
        return budget
    
    def enforce_budget(
        self,
        blocks: List[ContextBlock]
    ) -> Tuple[List[ContextBlock], Dict]:
        """
        Enforce token budget by trimming lowest-priority blocks.
        
        Args:
            blocks: List of context blocks
            
        Returns:
            Tuple of (trimmed_blocks, trimming_log)
        """
        trimming_log = {
            'initial_total': sum(b.tokens for b in blocks),
            'max_budget': self._max_context_tokens,
            'trimmed_blocks': [],
            'final_total': 0
        }
        
        # Check if we need trimming
        current_total = sum(b.tokens for b in blocks)
        if current_total <= self._max_context_tokens:
            trimming_log['final_total'] = current_total
            return blocks, trimming_log
        
        if not self._enable_trimming:
            logger.warning(f"Token budget exceeded ({current_total} > {self._max_context_tokens}) but trimming disabled")
            trimming_log['final_total'] = current_total
            return blocks, trimming_log
        
        # Sort blocks by priority (lower priority = trim first)
        # Create a mutable copy
        blocks_by_priority = sorted(
            blocks, 
            key=lambda b: (b.priority.value, b.priority_score)
        )
        
        # Keep CRITICAL and HIGH priority blocks, trim others
        protected = [b for b in blocks_by_priority if b.priority.value <= ContextPriority.HIGH.value]
        trimable = [b for b in blocks_by_priority if b.priority.value > ContextPriority.HIGH.value]
        
        # Calculate protected token count
        protected_tokens = sum(b.tokens for b in protected)
        available_for_trimable = self._max_context_tokens - protected_tokens - self._reserve_tokens
        
        logger.info(f"Token budget: {current_total} > {self._max_context_tokens}, trimming {len(trimable)} blocks")
        
        # Trim blocks starting from lowest priority
        trimmed_blocks = list(protected)
        current_trimmable = 0
        
        for block in trimable:
            if current_trimmable + block.tokens <= available_for_trimable:
                trimmed_blocks.append(block)
                current_trimmable += block.tokens
            else:
                # Partially trim this block
                if available_for_trimable > current_trimmable:
                    remaining_tokens = available_for_trimable - current_trimmable
                    # Keep portion of content proportional to remaining tokens
                    if block.content and remaining_tokens > 0:
                        ratio = remaining_tokens / block.tokens
                        chars_to_keep = int(len(block.content) * ratio)
                        trimmed_content = block.content[:chars_to_keep]
                        
                        trimmed_block = ContextBlock(
                            name=block.name,
                            content=trimmed_content,
                            tokens=remaining_tokens,
                            priority=block.priority,
                            priority_score=block.priority_score
                        )
                        trimmed_blocks.append(trimmed_block)
                        current_trimmable += remaining_tokens
                        
                        trimming_log['trimmed_blocks'].append({
                            'name': block.name,
                            'original_tokens': block.tokens,
                            'remaining_tokens': remaining_tokens,
                            'trimmed': True
                        })
                        
                        logger.info(f"Partially trimmed {block.name}: {block.tokens} -> {remaining_tokens} tokens")
                else:
                    # Fully trim this block
                    trimming_log['trimmed_blocks'].append({
                        'name': block.name,
                        'original_tokens': block.tokens,
                        'remaining_tokens': 0,
                        'trimmed': True
                    })
                    logger.info(f"Fully trimmed {block.name}: {block.tokens} tokens")
        
        # Reconstruct context from trimmed blocks
        trimmed_blocks.sort(key=lambda b: b.priority.value)
        trimming_log['final_total'] = sum(b.tokens for b in trimmed_blocks)
        
        # Log trimming event
        logger.info(f"Token trimming complete: {trimming_log['initial_total']} -> {trimming_log['final_total']} tokens")
        
        if self._debug_mode:
            logger.debug(f"Trimming log: {trimming_log}")
        
        return trimmed_blocks, trimming_log
    
    def get_trimmed_context(
        self,
        system_prompt: str,
        structured_memory: str,
        rolling_summary: str,
        semantic_memories: str,
        documents: str,
        short_term: str,
        user_query: str
    ) -> Tuple[Dict[str, str], TokenBudget, Dict]:
        """
        Get trimmed context within token budget.
        
        Args:
            All context strings
            
        Returns:
            Tuple of (context_dict, budget, trimming_log)
        """
        # Build blocks
        blocks = self.build_context_blocks(
            system_prompt=system_prompt,
            structured_memory=structured_memory,
            rolling_summary=rolling_summary,
            semantic_memories=semantic_memories,
            documents=documents,
            short_term=short_term,
            user_query=user_query
        )
        
        # Calculate budget
        budget = self.calculate_budget(
            system_prompt=system_prompt,
            structured_memory=structured_memory,
            rolling_summary=rolling_summary,
            semantic_memories=semantic_memories,
            documents=documents,
            short_term=short_term,
            user_query=user_query
        )
        
        # Enforce budget
        trimmed_blocks, trimming_log = self.enforce_budget(blocks)
        
        # Reconstruct context dict
        context_dict = {}
        for block in trimmed_blocks:
            context_dict[block.name] = block.content
        
        # Ensure all keys exist
        for key in ['system_prompt', 'structured', 'rolling_summary', 'semantic', 
                    'documents', 'short_term', 'user_query']:
            if key not in context_dict:
                context_dict[key] = ""
        
        return context_dict, budget, trimming_log
    
    # ================================================
    # PHASE 4: CONTEXT COMPRESSION METHODS
    # ================================================
    
    def compress_rolling_summary(
        self, 
        summary_text: str, 
        max_tokens: int = None
    ) -> CompressionResult:
        """
        Compress a rolling summary by extracting key points.
        
        Args:
            summary_text: The summary text to compress
            max_tokens: Maximum tokens after compression
            
        Returns:
            CompressionResult with compressed content
        """
        original_tokens = self.estimate_tokens(summary_text)
        
        if max_tokens is None:
            max_tokens = int(original_tokens * 0.5)  # Compress to 50%
        
        if original_tokens <= max_tokens:
            return CompressionResult(
                original_content=summary_text,
                compressed_content=summary_text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_type='summary'
            )
        
        # Extract key sentences (first and last, plus any bullet points)
        lines = summary_text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Keep bullet points, numbered items, and section headers
            if line.startswith(('-', '*', '•', '1.', '2.', '3.', '•')):
                key_points.append(line)
            # Keep section headers
            elif line.endswith(':') and len(line) < 50:
                key_points.append(line)
        
        # If we have key points, use them
        if key_points:
            compressed = '\n'.join(key_points)
            compressed_tokens = self.estimate_tokens(compressed)
            
            if compressed_tokens <= max_tokens:
                logger.info(f"Compressed rolling summary: {original_tokens} -> {compressed_tokens} tokens")
                return CompressionResult(
                    original_content=summary_text,
                    compressed_content=compressed,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    compression_type='summary'
                )
        
        # Fallback: truncate to max tokens
        chars_to_keep = int(max_tokens * self.CHARS_PER_TOKEN)
        compressed = summary_text[:chars_to_keep]
        
        logger.info(f"Truncated rolling summary: {original_tokens} -> {max_tokens} tokens")
        
        return CompressionResult(
            original_content=summary_text,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=max_tokens,
            compression_type='summary'
        )
    
    def compress_conversation(
        self, 
        conversation_text: str, 
        max_tokens: int = None
    ) -> CompressionResult:
        """
        Compress short-term conversation by keeping recent messages.
        
        Args:
            conversation_text: The conversation to compress
            max_tokens: Maximum tokens after compression
            
        Returns:
            CompressionResult with compressed content
        """
        original_tokens = self.estimate_tokens(conversation_text)
        
        if max_tokens is None:
            max_tokens = int(original_tokens * 0.5)  # Compress to 50%
        
        if original_tokens <= max_tokens:
            return CompressionResult(
                original_content=conversation_text,
                compressed_content=conversation_text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_type='conversation'
            )
        
        # Split into messages and keep most recent ones
        messages = conversation_text.split('\n\n')
        
        # Estimate tokens per message
        msg_tokens = [self.estimate_tokens(m) for m in messages]
        total = 0
        kept_messages = []
        
        # Keep messages from the end (most recent)
        for msg, tokens in zip(reversed(messages), reversed(msg_tokens)):
            if total + tokens <= max_tokens:
                kept_messages.insert(0, msg)
                total += tokens
            else:
                break
        
        # If we kept too few, try keeping every other message
        if len(kept_messages) < 2 and len(messages) > 2:
            kept_messages = []
            total = 0
            for msg, tokens in zip(messages, msg_tokens):
                if total + tokens <= max_tokens:
                    kept_messages.append(msg)
                    total += tokens
        
        compressed = '\n\n'.join(kept_messages)
        compressed_tokens = self.estimate_tokens(compressed)
        
        logger.info(f"Compressed conversation: {original_tokens} -> {compressed_tokens} tokens")
        
        return CompressionResult(
            original_content=conversation_text,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_type='conversation'
        )
    
    def compress_semantic_memories(
        self, 
        memories_text: str, 
        max_tokens: int = None
    ) -> CompressionResult:
        """
        Compress semantic memories by keeping highest-scoring ones.
        
        Since we can't re-score without the query, we keep the first N entries
        (which are typically the most recent/important).
        
        Args:
            memories_text: The memories text to compress
            max_tokens: Maximum tokens after compression
            
        Returns:
            CompressionResult with compressed content
        """
        original_tokens = self.estimate_tokens(memories_text)
        
        if max_tokens is None:
            max_tokens = int(original_tokens * 0.5)  # Compress to 50%
        
        if original_tokens <= max_tokens:
            return CompressionResult(
                original_content=memories_text,
                compressed_content=memories_text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_type='semantic'
            )
        
        # Split into individual memories (each starts with "- ")
        memory_lines = []
        current_memory = ""
        
        for line in memories_text.split('\n'):
            if line.startswith('- '):
                if current_memory:
                    memory_lines.append(current_memory)
                current_memory = line
            elif current_memory:
                current_memory += '\n' + line
        
        if current_memory:
            memory_lines.append(current_memory)
        
        # If no clear structure, just truncate
        if not memory_lines:
            chars_to_keep = int(max_tokens * self.CHARS_PER_TOKEN)
            compressed = memories_text[:chars_to_keep]
            
            return CompressionResult(
                original_content=memories_text,
                compressed_content=compressed,
                original_tokens=original_tokens,
                compressed_tokens=max_tokens,
                compression_type='semantic'
            )
        
        # Keep memories until we reach max_tokens
        kept_memories = []
        total = 0
        
        for mem in memory_lines:
            mem_tokens = self.estimate_tokens(mem)
            if total + mem_tokens <= max_tokens:
                kept_memories.append(mem)
                total += mem_tokens
            else:
                break
        
        compressed = '\n'.join(kept_memories)
        compressed_tokens = self.estimate_tokens(compressed)
        
        logger.info(f"Compressed semantic memories: {original_tokens} -> {compressed_tokens} tokens")
        
        return CompressionResult(
            original_content=memories_text,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_type='semantic'
        )
    
    def compress_context(
        self,
        blocks: List[ContextBlock],
        budget: TokenBudget
    ) -> Tuple[List[ContextBlock], Dict]:
        """
        Compress context blocks when budget is exceeded.
        
        This method is called when trimming alone isn't sufficient.
        It applies intelligent compression to reduce token count while
        preserving key information.
        
        Args:
            blocks: Current context blocks
            budget: Current token budget
            
        Returns:
            Tuple of (compressed_blocks, compression_log)
        """
        compression_log = {
            'compressions': [],
            'total_saved': 0
        }
        
        # Check if we need compression
        current_total = sum(b.tokens for b in blocks)
        available = self._max_context_tokens - self._reserve_tokens
        
        if current_total <= available:
            return blocks, compression_log
        
        # Find blocks that can be compressed
        for block in blocks:
            if block.name == 'rolling_summary' and block.tokens > 100:
                # Compress rolling summary
                result = self.compress_rolling_summary(block.content)
                if result.compressed_tokens < block.tokens:
                    block.content = result.compressed_content
                    block.tokens = result.compressed_tokens
                    block.compressed = True
                    block.original_tokens = result.original_tokens
                    compression_log['compressions'].append({
                        'block': block.name,
                        'type': 'summary',
                        'original': result.original_tokens,
                        'compressed': result.compressed_tokens
                    })
                    compression_log['total_saved'] += result.original_tokens - result.compressed_tokens
            
            elif block.name == 'short_term' and block.tokens > 200:
                # Compress conversation
                result = self.compress_conversation(block.content)
                if result.compressed_tokens < block.tokens:
                    block.content = result.compressed_content
                    block.tokens = result.compressed_tokens
                    block.compressed = True
                    block.original_tokens = result.original_tokens
                    compression_log['compressions'].append({
                        'block': block.name,
                        'type': 'conversation',
                        'original': result.original_tokens,
                        'compressed': result.compressed_tokens
                    })
                    compression_log['total_saved'] += result.original_tokens - result.compressed_tokens
            
            elif block.name == 'semantic' and block.tokens > 100:
                # Compress semantic memories
                result = self.compress_semantic_memories(block.content)
                if result.compressed_tokens < block.tokens:
                    block.content = result.compressed_content
                    block.tokens = result.compressed_tokens
                    block.compressed = True
                    block.original_tokens = result.original_tokens
                    compression_log['compressions'].append({
                        'block': block.name,
                        'type': 'semantic',
                        'original': result.original_tokens,
                        'compressed': result.compressed_tokens
                    })
                    compression_log['total_saved'] += result.original_tokens - result.compressed_tokens
        
        # Log compression events
        if compression_log['compressions']:
            logger.info(f"Context compression: saved {compression_log['total_saved']} tokens")
            for comp in compression_log['compressions']:
                logger.info(f"  - {comp['block']}: {comp['original']} -> {comp['compressed']} ({comp['type']})")
        
        return blocks, compression_log


# ============================================
# Config integration
# ============================================

class ConfigTokenBudget:
    """Config extension for token budget settings."""
    
    @property
    def max_context_tokens(self) -> int:
        """Maximum tokens for context (excluding generation)."""
        return int(self._config.get('token_budget', {}).get('max_context_tokens', 32000))
    
    @property
    def token_reserve_for_generation(self) -> int:
        """Reserve tokens for generation output."""
        return int(self._config.get('token_budget', {}).get('reserve_for_generation', 2048))
    
    @property
    def token_trimming_enabled(self) -> bool:
        """Enable automatic token trimming."""
        return bool(self._config.get('token_budget', {}).get('enable_trimming', True))
    
    @property
    def token_estimation_method(self) -> str:
        """Token estimation method: 'chars' or 'accurate'."""
        return self._config.get('token_budget', {}).get('estimation_method', 'chars')

