"""
OpenRouter LLM Client - OpenAI-compatible wrapper for Nemotron via OpenRouter.
Replaces DeepSeek with nvidia/nemotron-3-super-120b-a12b:free (free tier).
"""

import json
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:
    raise ImportError(\"OpenAI SDK required: pip install openai\")


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    \"\"\"Structured response from OpenRouter LLM.\"\"\"
    content: str
    raw_message: Optional[dict] = None


class OpenRouterClient:
    \"\"\"
    OpenAI-compatible client for OpenRouter API using Nemotron model.
    Supports retry logic with exponential backoff.
    \"\"\"

    MAX_RETRIES = 3
    BASE_BACKOFF = 1.0

    def __init__(
        self,
        api_key: str,
        model: str = \"nvidia/nemotron-3-super-120b-a12b:free\",
        base_url: str = \"https://openrouter.ai/api/v1\",
        timeout: int = 120,
        max_tokens: int = 8000,
    ):
        \"\"\"
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (https://openrouter.ai/keys)
            model: Model name (default: Nemotron free)
            base_url: OpenRouter API base
            timeout: Request timeout
            max_tokens: Max response tokens
        \"\"\"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        logger.info(f\"OpenRouter client initialized: {model}\")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ) -> LLMResponse:
        \"\"\"
        Send message with retry logic.

        Args:
            messages: Chat messages list
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            max_retries: Retry attempts

        Returns:
            LLMResponse with content

        Raises:
            Exception after all retries fail
        \"\"\"
        formatted_messages = messages.copy()
        if system_prompt:
            formatted_messages.insert(0, {\"role\": \"system\", \"content\": system_prompt})

        for attempt in range(max_retries):
            try:
                logger.debug(f\"OpenRouter chat attempt {attempt + 1}/{max_retries}\")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )

                message = response.choices[0].message
                content = message.content or \"\"

                logger.debug(\"OpenRouter response received\")
                return LLMResponse(
                    content=content,
                    raw_message=message.model_dump() if hasattr(message, \"model_dump\") else None,
                )

            except (RateLimitError, APIError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    logger.error(f\"OpenRouter final error after {max_retries} retries: {e}\")
                    raise

                backoff = self.BASE_BACKOFF * (2 ** attempt)
                logger.warning(f\"OpenRouter retry {attempt + 1}/{max_retries} in {backoff:.1f}s: {e}\")
                time.sleep(backoff)

            except Exception as e:
                logger.error(f\"Unexpected OpenRouter error: {e}\")
                raise

        raise Exception(\"All retries exhausted\")

    def get_action_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        \"\"\"
        Parse JSON action from response.
        \"\"\"
        response = self.chat(messages, system_prompt=system_prompt)
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        \"\"\"
        Extract and parse JSON from response text.
        \"\"\"
        text = text.strip()
        if text.startswith(\"```json\"):
            text = text.split(\"```json\")[1].split(\"```\")[0].strip()
        elif text.startswith(\"```\"):
            lines = text.split(\"\\n\")
            json_lines = [line for line in lines if not line.startswith(\"```\")]
            text = \"\\n\".join(json_lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f\"JSON parse failed: {e} | Text: {text[:200]}\")
            raise ValueError(f\"Invalid JSON response: {e}\")

    def test_connection(self) -> bool:
        \"\"\"
        Test API connectivity.
        \"\"\"
        try:
            response = self.chat([{\"role\": \"user\", \"content\": \"OK\"}], system_prompt=\"Respond OK.\")
            logger.info(\"OpenRouter connection test OK\")
            return bool(response.content and \"OK\" in response.content)
        except Exception as e:
            logger.error(f\"Connection test failed: {e}\")
            return False

