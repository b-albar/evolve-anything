"""
LLM Models module.

This module provides abstract base classes and concrete implementations
for LLM and embedding providers using OpenRouter's unified API.
"""

from .base import BaseLLMProvider, BaseEmbeddingProvider
from .openrouter import OpenRouterProvider
from .openrouter_embedding import OpenRouterEmbeddingProvider
from .result import QueryResult

# Reasoning models - models that support extended thinking/reasoning
REASONING_MODELS = [
    # OpenAI reasoning models
    "openai/o1-2024-12-17",
    "openai/o3-mini-2025-01-31",
    "openai/o3-mini",
    "openai/o3-2025-04-16",
    "openai/o4-mini-2025-04-16",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    # Anthropic reasoning models
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.7-sonnet-20250219",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-4-sonnet",
    "anthropic/claude-sonnet-4-5",
    # DeepSeek reasoning models
    "deepseek/deepseek-reasoner",
    # Gemini reasoning models
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-3-pro-preview",
]

# Thinking tokens configuration for reasoning models
THINKING_TOKENS = {
    "auto": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}

__all__ = [
    # Abstract base classes
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    # Concrete providers
    "OpenRouterProvider",
    "OpenRouterEmbeddingProvider",
    # Data classes
    "QueryResult",
    # Model configurations
    "REASONING_MODELS",
    "THINKING_TOKENS",
]
