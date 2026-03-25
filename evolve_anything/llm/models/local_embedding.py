"""
Local Embedding Provider using sentence-transformers.

Loads HuggingFace models locally via sentence-transformers, avoiding
the need for an external embedding API.
"""

import logging
from typing import List, Optional, Tuple, Union

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Loads a HuggingFace model locally and runs inference on-device.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/CodeRankEmbed",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.verbose = verbose
        self._model = None

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            if self.verbose:
                logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def get_embedding(
        self, texts: Union[str, List[str]]
    ) -> Tuple[Union[List[float], List[List[float]]], int]:
        """
        Get embeddings for text(s) using the local model.

        Returns:
            Tuple of (embeddings, 0). Token count is always 0 for local models.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        result = [emb.tolist() for emb in embeddings]

        if single_input:
            return result[0] if result else [], 0
        return result, 0
