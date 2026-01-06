"""
Abstract base class for LLM providers.

This module defines the base interface that all LLM providers must implement.
This allows for easy extension to support future providers while maintaining
a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from .result import QueryResult


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure consistent
    behavior across different backends.
    """

    @abstractmethod
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    def get_client(self, structured_output: bool = False) -> Any:
        """
        Get the client instance for making API calls.

        Args:
            structured_output: Whether to enable structured output mode

        Returns:
            The client instance
        """
        pass

    @abstractmethod
    def query(
        self,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: List[Dict],
        output_model: Optional[BaseModel] = None,
        model_posteriors: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a query to the LLM.

        Args:
            model: The model identifier to use
            msg: The user message
            system_msg: The system prompt
            msg_history: Previous conversation history
            output_model: Optional Pydantic model for structured output
            model_posteriors: Optional model posteriors for logging
            **kwargs: Additional query parameters

        Returns:
            QueryResult containing the response and metadata
        """
        pass


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different backends.
    """

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the embedding provider.

        Args:
            model_name: The embedding model to use
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    def get_embedding(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (embeddings list, total tokens used)
        """
        pass
