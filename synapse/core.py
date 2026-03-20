"""SYNAPSE Core - Central orchestrator for the AI assistant."""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from .llm import OpenRouterClient
from .tools import ToolRegistry
from .memory import ConversationMemory


logger = logging.getLogger(__name__)


class SynapseCore:
    """
    Central orchestrator for SYNAPSE AI assistant.
    Handles user input, LLM communication, tool execution, and memory.
    """

    def __init__(
        self,
        openrouter_api_key: str,
        system_prompt: Optional[str] = None,
        memory_max_messages: int = 20,
        persist_memory: bool = False,
        debug: bool = False,
    ):
        """
        Initialize SYNAPSE Core.

        Args:
            openrouter_api_key: OpenRouter API key (https://openrouter.ai/keys)
            system_prompt: Custom system prompt
            memory_max_messages: Max messages in conversation history
            persist_memory: Save/load memory from disk
            debug: Enable debug logging
        """
        self.debug = debug

        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(level=logging.INFO)

        # Initialize components
        self.llm = OpenRouterClient(openrouter_api_key=openrouter_api_key)
        self.tools = ToolRegistry()
        self.memory = ConversationMemory(
            max_messages=memory_max_messages,
            persist_path="synapse_memory.json" if persist_memory else None,
        )

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        logger.info("SYNAPSE Core initialized")

    def _default_system_prompt(self) -> str:
        """Generate default system prompt with tool descriptions."""
        tools_schema = json.dumps(self.tools.get_tools_json_schema(), indent=2)
        return f"""You are SYNAPSE, a powerful AI assistant.

You have access to the following tools:

{self.tools.get_tools_description()}

When you need to execute an action, respond with a JSON object like this:
{{
  "action": "tool_name",
  "parameters": {{"param1": "value1", "param2": "value2"}},
  "reasoning": "Why you're doing this"
}}

Always provide clear explanations for your actions.
Be concise and helpful."""

    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Process user query end-to-end.

        Args:
            user_input: User's input message

        Returns:
            Dict with:
                - response: Final response to user
                - thinking: LLM thinking (if available)
                - action: Tool action executed (if any)
                - action_result: Result from tool (if any)
                - complete: Whether query was fully processed
        """
        logger.info(f"Processing query: {user_input[:100]}...")

        # Add user message to memory
        self.memory.add_message("user", user_input)

        # Get LLM response
        try:
            llm_response = self.llm.chat(
                messages=self.memory.get_messages(),
                system_prompt=self.system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.memory.add_message("assistant", error_msg)
            return {
                "response": error_msg,
                "thinking": None,
                "action": None,
                "action_result": None,
                "complete": False,
            }

        response_text = llm_response.content

        logger.debug(f"LLM response: {response_text[:200]}...")
        logger.debug(f"LLM response: {response_text[:200]}...")

        # Try to parse as JSON action
        action_name = None
        action_result = None
        final_response = response_text

        try:
            action_json = self.llm.get_action_json([{"role": "user", "content": response_text}], system_prompt=self.system_prompt)

            if "action" in action_json:
                action_name = action_json["action"]
                parameters = action_json.get("parameters", {})

                logger.info(f"Executing action: {action_name} with {parameters}")

                # Execute tool
                try:
                    action_result = self.tools.execute(action_name, parameters)
                    logger.info(f"Action result: {action_result[:100]}...")

                    # Add assistant message with action
                    self.memory.add_message(
                        "assistant",
                        response_text,
                        action=action_name,
                    )

                    # Get follow-up response from LLM about action result
                    followup_prompt = f"Action '{action_name}' executed with result: {action_result}\n\nProvide a brief response to the user based on this result."

                    self.memory.add_message("user", followup_prompt)
                    followup_response = self.llm.chat(
                        messages=self.memory.get_messages(),
                        system_prompt=self.system_prompt,
                    )

                    final_response = followup_response.content
                    self.memory.add_message("assistant", final_response)

                except ValueError as e:
                    logger.error(f"Unknown action: {action_name}: {e}")
                    error_msg = f"Unknown action: {action_name}"
                    final_response = error_msg
                    self.memory.add_message("assistant", error_msg)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    error_msg = f"Error executing action: {str(e)}"
                    final_response = error_msg
                    self.memory.add_message("assistant", error_msg)
            else:
                # Regular response, not an action
                self.memory.add_message("assistant", response_text)

        except ValueError:
            # Not JSON - regular response
            logger.debug("Response is not JSON, treating as regular response")
            self.memory.add_message("assistant", response_text)

        return {
            "response": final_response,
            "thinking": None,
            "action": action_name,
            "action_result": action_result,
            "complete": True,
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get conversation memory summary."""
        return self.memory.get_summary()

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

    def test_connection(self) -> bool:
        """Test DeepSeek API connection."""
        return self.llm.test_connection()
