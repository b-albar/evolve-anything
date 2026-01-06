"""
LLM module.

Provides LLM and embedding clients using OpenRouter's unified API.
"""

from .llm import LLMClient, extract_between
from .embedding import EmbeddingClient
from .models import (
    QueryResult,
    BaseLLMProvider,
    BaseEmbeddingProvider,
    OpenRouterProvider,
    OpenRouterEmbeddingProvider,
    REASONING_MODELS,
)
from .dynamic_sampling import (
    BanditBase,
    AsymmetricUCB,
    FixedSampler,
)

__all__ = [
    # Main clients
    "LLMClient",
    "EmbeddingClient",
    # Query result
    "QueryResult",
    # Abstract base classes (for implementing new providers)
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    # Concrete providers
    "OpenRouterProvider",
    "OpenRouterEmbeddingProvider",
    # Utility functions
    "extract_between",
    # Model configurations
    "REASONING_MODELS",
    # Dynamic sampling
    "BanditBase",
    "AsymmetricUCB",
    "FixedSampler",
]
