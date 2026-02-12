"""
Query module.

Provides functions for querying LLMs and sampling model kwargs.
Uses the OpenRouter provider for all model queries, with optional
support for Claude Code wrapper via model prefix routing.
"""

import os
from typing import List, Union, Optional, Dict, Type, Any
import random
from pydantic import BaseModel
import logging

from .models import (
    QueryResult,
    OpenRouterProvider,
    REASONING_MODELS,
)

logger = logging.getLogger(__name__)

# Global provider instances (lazy initialized)
_default_provider: OpenRouterProvider = None
_claude_code_provider: OpenRouterProvider = None

# Claude Code wrapper configuration
CLAUDE_CODE_PREFIX = "claude-code/"
CLAUDE_CODE_BASE_URL = os.getenv("CLAUDE_CODE_BASE_URL", "http://localhost:7999/v1")


def _get_provider(model_name: Optional[str] = None) -> OpenRouterProvider:
    """Get or create the appropriate provider instance based on model prefix.

    Args:
        model_name: The model name to determine which provider to use.
                   Models prefixed with 'claude-code/' use the Claude Code wrapper.

    Returns:
        The appropriate OpenRouterProvider instance.
    """
    global _default_provider, _claude_code_provider

    # Check if this is a Claude Code model
    if model_name and model_name.startswith(CLAUDE_CODE_PREFIX):
        if _claude_code_provider is None:
            # Use a dummy API key for Claude Code wrapper (it uses CLI auth)
            api_key = os.getenv("CLAUDE_CODE_API_KEY", "claude-code")
            _claude_code_provider = OpenRouterProvider(
                api_key=api_key,
                base_url=CLAUDE_CODE_BASE_URL,
            )
        return _claude_code_provider

    # Default provider for OpenRouter
    if _default_provider is None:
        _default_provider = OpenRouterProvider()
    return _default_provider


def _normalize_model_name(model_name: str) -> str:
    """Strip the claude-code/ prefix for the actual API call.

    The Claude Code wrapper accepts any model name, so we strip the prefix
    and pass the rest (e.g., 'claude-code/opus' -> 'opus').
    """
    if model_name.startswith(CLAUDE_CODE_PREFIX):
        return model_name[len(CLAUDE_CODE_PREFIX):]
    return model_name


def sample_model_kwargs(
    model_names: Union[List[str], str] = "openai/gpt-4o-mini",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
) -> Dict:
    """
    Sample a single kwargs dictionary for a model query.

    Args:
        model_names: Model name(s) to sample from
        temperatures: Temperature value(s) to sample from
        max_tokens: Max token value(s) to sample from
        reasoning_efforts: Reasoning effort level(s) to sample from
        model_sample_probs: Probability weights for model sampling

    Returns:
        A kwargs dictionary for query()
    """
    # Normalize all inputs to lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, float):
        temperatures = [temperatures]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    if isinstance(reasoning_efforts, str):
        reasoning_efforts = [reasoning_efforts]

    kwargs_dict: dict[str, Any] = {}

    # Sample model
    if model_sample_probs is not None:
        if len(model_sample_probs) != len(model_names):
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        kwargs_dict["model_name"] = random.choices(
            model_names, weights=model_sample_probs, k=1
        )[0]
    else:
        kwargs_dict["model_name"] = random.choice(model_names)

    model = kwargs_dict["model_name"]

    # Reasoning models always use temperature 1.0
    if model in REASONING_MODELS:
        kwargs_dict["temperature"] = 1.0
    else:
        kwargs_dict["temperature"] = random.choice(temperatures)

    # Set max tokens and reasoning parameters (uniform format)
    kwargs_dict["max_tokens"] = random.choice(max_tokens)

    if model in REASONING_MODELS:
        r_effort = random.choice(reasoning_efforts)
        if r_effort and r_effort != "auto":
            kwargs_dict["reasoning_effort"] = r_effort

    return kwargs_dict


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """
    Query the LLM using the appropriate provider.

    Args:
        model_name: The model to query. Supports:
            - OpenRouter models: "openai/gpt-4o", "anthropic/claude-3-5-sonnet"
            - Claude Code wrapper: "claude-code/opus", "claude-code/sonnet"
        msg: The user message
        system_msg: The system prompt
        msg_history: Previous conversation history
        output_model: Optional Pydantic model for structured output
        model_posteriors: Optional model posteriors for logging
        **kwargs: Additional query parameters (temperature, max_tokens, etc.)

    Returns:
        QueryResult containing the response and metadata
    """
    provider = _get_provider(model_name)
    api_model_name = _normalize_model_name(model_name)

    result = provider.query(
        model=api_model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        output_model=output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )

    # Preserve original model name in result for tracking
    result.model_name = model_name

    return result


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """
    Query the LLM asynchronously.

    Args:
        model_name: The model to query. Supports:
            - OpenRouter models: "openai/gpt-4o", "anthropic/claude-3-5-sonnet"
            - Claude Code wrapper: "claude-code/opus", "claude-code/sonnet"
        msg: The user message
        system_msg: The system prompt
        msg_history: Previous conversation history
        output_model: Optional Pydantic model for structured output
        model_posteriors: Optional model posteriors for logging
        **kwargs: Additional query parameters

    Returns:
        QueryResult containing the response and metadata
    """
    provider = _get_provider(model_name)
    api_model_name = _normalize_model_name(model_name)

    result = await provider.query_async(
        model=api_model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        output_model=output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )

    # Preserve original model name in result for tracking
    result.model_name = model_name

    return result
