"""
OpenRouter LLM Provider.

This provider implements the OpenAI-compatible API through OpenRouter,
allowing access to multiple model providers through a single interface.
All models are accessed using OpenAI-style API calls.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Type

import backoff
import openai
from pydantic import BaseModel

from .base import BaseLLMProvider
from .result import QueryResult

logger = logging.getLogger(__name__)


def backoff_handler(details):
    """Log backoff retry information."""
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenRouter - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM Provider.

    Uses OpenRouter's OpenAI-compatible API to access multiple model providers
    through a unified interface. Supports both sync and async operations.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key. Defaults to OPENAI_API_KEY env var.
            base_url: Base URL for the API. Defaults to OpenRouter's API URL.
            **kwargs: Additional configuration options
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None
        self._async_client = None

    def get_client(self, structured_output: bool = False) -> Any:
        """Get the sync OpenAI-compatible client."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def get_async_client(self) -> Any:
        """Get the async OpenAI-compatible client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._async_client

    def _pydantic_to_json_schema(self, model: Type[BaseModel]) -> Dict:
        """Convert a Pydantic model to JSON schema for OpenAI structured output."""
        schema = model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    def _prepare_kwargs(self, model: str, kwargs: Dict) -> Dict:
        """
        Translate uniform kwargs to model-specific format.

        Args:
            model: The model identifier (e.g., "openai/gpt-4o")
            kwargs: Uniform kwargs dict with max_tokens, reasoning_effort, etc.

        Returns:
            Model-specific kwargs dict ready for the API call
        """
        result = kwargs.copy()

        reasoning_effort = result.pop("reasoning_effort", None)
        if reasoning_effort:
            result["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_effort,
                }
            }

        return result

    def _build_result(
        self,
        response,
        msg: str,
        system_msg: str,
        new_msg_history: List[Dict],
        model: str,
        kwargs: Dict,
        output_model: Optional[Type[BaseModel]],
        model_posteriors: Optional[Dict[str, float]],
    ) -> QueryResult:
        """Build QueryResult from API response."""
        thought = ""

        if output_model is not None:
            raw_content = response.choices[0].message.content
            try:
                parsed_data = json.loads(raw_content)
                content = output_model(**parsed_data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse structured output: {e}")
                content = raw_content
        else:
            content = response.choices[0].message.content
            if hasattr(response.choices[0].message, "reasoning_content"):
                thought = response.choices[0].message.reasoning_content or ""

        new_msg_history.append({"role": "assistant", "content": str(content)})

        input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else 0
        output_tokens = (
            response.usage.completion_tokens if hasattr(response, "usage") else 0
        )

        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=new_msg_history,
            model_name=model,
            kwargs=kwargs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought=thought,
            model_posteriors=model_posteriors,
        )

    @backoff.on_exception(
        backoff.expo,
        (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.RateLimitError,
            openai.APITimeoutError,
        ),
        max_tries=20,
        max_value=20,
        on_backoff=backoff_handler,
    )
    def query(
        self,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: List[Dict],
        output_model: Optional[Type[BaseModel]] = None,
        model_posteriors: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a synchronous query to the LLM via OpenRouter.

        Args:
            model: The model identifier (e.g., "openai/gpt-4o")
            msg: The user message
            system_msg: The system prompt
            msg_history: Previous conversation history
            output_model: Optional Pydantic model for structured output
            model_posteriors: Optional model posteriors for logging
            **kwargs: Additional query parameters (temperature, max_tokens, etc.)

        Returns:
            QueryResult containing the response and metadata
        """
        client = self.get_client()
        new_msg_history = msg_history + [{"role": "user", "content": msg}]

        messages = [
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ]

        api_kwargs = self._prepare_kwargs(model, kwargs)

        if output_model is not None:
            response_format = self._pydantic_to_json_schema(output_model)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                **api_kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **api_kwargs,
            )

        return self._build_result(
            response,
            msg,
            system_msg,
            new_msg_history,
            model,
            kwargs,
            output_model,
            model_posteriors,
        )

    async def query_async(
        self,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: List[Dict],
        output_model: Optional[Type[BaseModel]] = None,
        model_posteriors: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute an asynchronous query to the LLM via OpenRouter.

        Uses true async I/O for efficient concurrent requests.

        Args:
            model: The model identifier (e.g., "openai/gpt-4o")
            msg: The user message
            system_msg: The system prompt
            msg_history: Previous conversation history
            output_model: Optional Pydantic model for structured output
            model_posteriors: Optional model posteriors for logging
            **kwargs: Additional query parameters (temperature, max_tokens, etc.)

        Returns:
            QueryResult containing the response and metadata
        """
        client = self.get_async_client()
        new_msg_history = msg_history + [{"role": "user", "content": msg}]

        messages = [
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ]

        api_kwargs = self._prepare_kwargs(model, kwargs)

        if output_model is not None:
            response_format = self._pydantic_to_json_schema(output_model)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                **api_kwargs,
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **api_kwargs,
            )

        return self._build_result(
            response,
            msg,
            system_msg,
            new_msg_history,
            model,
            kwargs,
            output_model,
            model_posteriors,
        )
