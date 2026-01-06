"""
OpenRouter Embedding Provider.

This provider implements the OpenAI-compatible embedding API through OpenRouter,
allowing access to multiple embedding model providers through a single interface.
"""

import os
import logging
from typing import List, Optional, Tuple, Union

import backoff
import openai

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def backoff_handler(details):
    """Log backoff retry information."""
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenRouter Embedding - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenRouter Embedding Provider.

    Uses OpenRouter's OpenAI-compatible API to access multiple embedding model
    providers through a unified interface.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenRouter embedding provider.

        Args:
            model_name: The embedding model to use (e.g., "openai/text-embedding-3-small")
            api_key: OpenRouter API key. Defaults to OPENAI_API_KEY env var.
            base_url: Base URL for the API. Defaults to OpenRouter's API URL.
            verbose: Whether to enable verbose logging
            **kwargs: Additional configuration options
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name
        self.verbose = verbose
        self._client = None

    def _get_client(self) -> openai.OpenAI:
        """Get or create the OpenAI-compatible client."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @backoff.on_exception(
        backoff.expo,
        (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.RateLimitError,
            openai.APITimeoutError,
        ),
        max_tries=10,
        max_value=20,
        on_backoff=backoff_handler,
    )
    def get_embedding(
        self, texts: Union[str, List[str]]
    ) -> Tuple[Union[List[float], List[List[float]]], int]:
        """
        Get embeddings for text(s).

        Args:
            texts: A single text string or list of text strings to embed

        Returns:
            Tuple of (embeddings, total_tokens). If single text was provided,
            returns (single_embedding, tokens). If list was provided,
            returns (list_of_embeddings, tokens).
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
            )

            embeddings = [data.embedding for data in response.data]
            total_tokens = response.usage.total_tokens

            if single_input:
                return embeddings[0] if embeddings else [], total_tokens
            return embeddings, total_tokens

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            if single_input:
                return [], 0
            return [[]], 0
