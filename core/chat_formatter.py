"""
ChatFormatter module for handling model-specific chat template formatting.
Supports Qwen2.5 and other chat-based models with proper role handling.
"""

from typing import List, Dict, Optional
from enum import Enum


class Role(str, Enum):
    """Enum for chat roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ModelType(str, Enum):
    """Enum for model types with specific formatting requirements."""
    QWEN = "qwen"
    QWEN2_5 = "qwen2.5"
    MISTRAL = "mistral"
    DEFAULT = "default"


class Message:
    """Represents a single message in the conversation."""
    
    def __init__(self, role: Role, content: str):
        """
        Initialize a Message.
        
        Args:
            role: The role of the message sender (system, user, or assistant)
            content: The text content of the message
        """
        self.role = role if isinstance(role, Role) else Role(role)
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role.value, "content": self.content}


class ChatFormatter:
    """
    Base class for chat formatting with model-aware template support.
    Handles conversion of structured conversation to model-specific formats.
    """
    
    def __init__(self, model_type: ModelType = ModelType.QWEN2_5):
        """
        Initialize ChatFormatter.
        
        Args:
            model_type: The type of model to format for (defaults to Qwen2.5)
        """
        self.model_type = model_type if isinstance(model_type, ModelType) else ModelType(model_type)
    
    def format_prompt(self, messages: List[Message]) -> str:
        """
        Format a list of messages into a prompt string for the model.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Formatted prompt string
        """
        if self.model_type in [ModelType.QWEN, ModelType.QWEN2_5]:
            return self._format_qwen_prompt(messages)
        elif self.model_type == ModelType.MISTRAL:
            return self._format_mistral_prompt(messages)
        else:
            return self._format_default_prompt(messages)
    
    def _format_qwen_prompt(self, messages: List[Message]) -> str:
        """
        Format messages using Qwen2.5 chat template.
        
        Qwen2.5 uses: <|im_start|>role\ncontent\n<|im_end|>
        """
        formatted_parts = []
        
        for message in messages:
            formatted_parts.append(f"<|im_start|>{message.role.value}\n{message.content}\n<|im_end|>")
        
        # Add assistant opening for generation
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)
    
    def _format_mistral_prompt(self, messages: List[Message]) -> str:
        """
        Format messages using Mistral format.
        Uses [INST] tags for user/system instructions.
        """
        formatted_parts = []
        
        for message in messages:
            if message.role == Role.SYSTEM:
                formatted_parts.append(f"[SYSTEM]\n{message.content}")
            elif message.role == Role.USER:
                formatted_parts.append(f"[INST]{message.content}[/INST]")
            elif message.role == Role.ASSISTANT:
                formatted_parts.append(message.content)
        
        return "\n".join(formatted_parts)
    
    def _format_default_prompt(self, messages: List[Message]) -> str:
        """
        Format messages using default conversational format.
        """
        formatted_parts = []
        
        for message in messages:
            formatted_parts.append(f"{message.role.value.capitalize()}: {message.content}")
        
        return "\n".join(formatted_parts) + "\nAssistant: "
    
    def build_prompt(
        self,
        system: str,
        user: str,
        history: Optional[str] = None,
        include_assistant_prefix: bool = True
    ) -> str:
        """
        Convenience method to build a prompt from components.
        
        Args:
            system: System message/instructions
            user: Current user message
            history: Optional conversation history
            include_assistant_prefix: Whether to add assistant prefix for generation
            
        Returns:
            Formatted prompt ready for model generation
        """
        messages = []
        
        # Add system message
        if system:
            messages.append(Message(Role.SYSTEM, system))
        
        # Add history if provided
        if history:
            messages.append(Message(Role.USER, history))
        
        # Add current user message
        messages.append(Message(Role.USER, user))
        
        # Format for generation
        if self.model_type in [ModelType.QWEN, ModelType.QWEN2_5]:
            formatted = self._format_qwen_prompt(messages[:-1])  # Format all but last
            if include_assistant_prefix:
                formatted += f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted += f"<|im_start|>user\n{user}\n<|im_end|>"
            return formatted
        else:
            messages_for_format = messages if not include_assistant_prefix else messages
            return self.format_prompt(messages_for_format)
    
    def extract_response(self, raw_output: str) -> str:
        """
        Extract the assistant's response from raw model output.
        Handles differences in model output formatting.
        
        Args:
            raw_output: Raw text output from the model
            
        Returns:
            Cleaned response text
        """
        # Remove end-of-message markers
        if "<|im_end|>" in raw_output:
            raw_output = raw_output.split("<|im_end|>")[0]
        
        # Strip whitespace
        return raw_output.strip()
