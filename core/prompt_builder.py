"""
PromptBuilder module - Builds prompts in a modular, model-agnostic way.
Accepts Config via constructor for strict config usage.
"""

from typing import Dict, List
from core.config import Config


class PromptBuilder:
    """
    Builds prompts in a modular, model-agnostic way.
    
    The builder returns two top-level strings expected by `ModelManager.format_chat`:
    - system: system instructions
    - user: the user-facing prompt (includes retrieved docs, memories, recent convo, and query)
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize PromptBuilder with optional config.
        
        Args:
            config: Config instance for settings (optional, uses singleton if not provided)
        """
        self._config = config or Config()
    
    @staticmethod
    def build(
        system_prompt: str, 
        retrieved_docs: List[Dict], 
        memories: Dict[str, str], 
        recent_conv: str, 
        user_message: str
    ) -> Dict[str, str]:
        """
        Build prompt components.
        
        Args:
            system_prompt: System instructions
            retrieved_docs: List of retrieved document chunks
            memories: Dict with 'semantic', 'rolling_summary', 'structured' keys
            recent_conv: Recent conversation context
            user_message: Current user message
            
        Returns:
            Dict with 'system' and 'user' keys
        """
        parts = []

        # Retrieved documents block (kept out of memory stores)
        if retrieved_docs:
            doc_lines = ["=== Retrieved Document Context ==="]
            for d in retrieved_docs:
                # each d is expected to have 'chunk' with metadata and 'text'
                chunk = d.get('chunk', {})
                meta = f"Document: {chunk.get('doc_name')} | Page: {chunk.get('page_number')} | Chunk: {chunk.get('chunk_index')}"
                doc_lines.append(f"{meta}\n{chunk.get('text')}")
            parts.append("\n\n".join(doc_lines))

        # Long-term semantic memory
        if memories.get('semantic'):
            parts.append(memories['semantic'])

        # Rolling summary
        if memories.get('rolling_summary'):
            parts.append(memories['rolling_summary'])

        # Structured
        if memories.get('structured'):
            parts.append(memories['structured'])

        # Recent conversation
        if recent_conv:
            parts.append(f"Current Conversation:\n{recent_conv}")

        # User query at the end
        parts.append(f"User Query:\n{user_message}")

        # Join into a single user-facing block
        user_block = "\n\n".join(p for p in parts if p)

        return {
            'system': system_prompt or '',
            'user': user_block
        }

