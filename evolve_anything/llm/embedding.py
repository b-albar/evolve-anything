"""
Embedding module.

Provides embedding functionality using OpenRouter's unified API.
"""

import pandas as pd
from typing import Union, List, Tuple
import numpy as np
import logging

from .models import OpenRouterEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Embedding client using OpenAI-compatible API.

    This client provides a simplified interface for generating embeddings
    across multiple providers.
    """

    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-small",
        verbose: bool = False,
    ):
        """
        Initialize the EmbeddingClient.

        Args:
            model_name: The embedding model to use (e.g., "openai/text-embedding-3-small")
            verbose: Whether to enable verbose logging
        """
        self.model_name = model_name
        self.verbose = verbose
        self._provider = OpenRouterEmbeddingProvider(
            model_name=self.model_name,
            verbose=verbose,
        )

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Union[Tuple[List[float], int], Tuple[List[List[float]], int]]:
        """
        Computes the text embedding for a string or list of strings.

        Args:
            code: The text as a string or list of strings to embed.

        Returns:
            Tuple of (embedding(s), token_count). If single string was provided,
            returns (single_embedding, tokens). If list was provided,
            returns (list_of_embeddings, tokens).
        """
        return self._provider.get_embedding(code)

    def get_column_embedding(
        self,
        df: pd.DataFrame,
        column_name: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Computes the text embedding for a column in a DataFrame.

        Args:
            df: A pandas DataFrame with the column(s) to embed.
            column_name: The name of the column(s) to embed.

        Returns:
            pd.DataFrame: The DataFrame with new embedding columns added.
        """
        if isinstance(column_name, str):
            column_name = [column_name]

        for col in column_name:
            model_name_str = self.model_name.replace("/", "_").replace("-", "_")
            new_col_name = f"{col}_embedding_{model_name_str}"
            df[new_col_name] = df[col].apply(
                lambda x: self.get_embedding(x),
            )
        return df

    def get_closest_k_neighbors(
        self,
        new_str_query: str,
        embeddings: list,
        top_k: Union[int, str] = 5,
    ) -> Tuple[list, list]:
        """
        Get k closest neighbors from the embeddings list.

        Args:
            new_str_query: The string to get the closest neighbors for.
            embeddings: The list of embeddings to compare against.
            top_k: The number of closest neighbors to return, or "random".

        Returns:
            A tuple of (top_k_indices, top_k_similarities).
        """
        new_embedding, _ = self.get_embedding(new_str_query)

        if not new_embedding:
            return [], []

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = [
            cosine_similarity(new_embedding, embedding) for embedding in embeddings
        ]

        if top_k == "random":
            if len(similarities) < 5:
                top_idx = np.random.choice(
                    len(similarities), size=len(similarities), replace=False
                )
            else:
                top_idx = np.random.choice(len(similarities), size=5, replace=False)
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx.tolist(), similarities_subset
        elif isinstance(top_k, int):
            top_idx = np.argsort(similarities)[-top_k:]
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx[::-1].tolist(), similarities_subset[::-1]
        else:
            raise ValueError("top_k must be an int or 'random'")

    def get_dim_reduction(
        self,
        embeddings: list,
        method: str = "pca",
        dims: int = 2,
    ):
        """
        Performs dimensionality reduction on a list of embeddings.

        Args:
            embeddings: List of embedding vectors
            method: Dimensionality reduction method ('pca', 'umap', or 'tsne')
            dims: Number of dimensions to reduce to

        Returns:
            The transformed embeddings in reduced dimensionality
        """
        if isinstance(embeddings, pd.Series):
            embeddings = embeddings.tolist()

        X = np.array(embeddings) if isinstance(embeddings, list) else embeddings

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if method.lower() == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=dims)
            return model.fit_transform(X)
        elif method.lower() == "umap":
            from umap import UMAP

            model = UMAP(n_components=dims, random_state=42)
            return model.fit_transform(X)
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE

            model = TSNE(n_components=dims, random_state=42)
            return model.fit_transform(X)
        else:
            raise ValueError("Method must be one of: 'pca', 'umap', 'tsne'")

    def get_embedding_clusters(
        self,
        embeddings: list,
        num_clusters: int = 4,
        verbose: bool = False,
    ) -> list:
        """
        Performs clustering on a list of embeddings using Gaussian Mixture Model.

        Args:
            embeddings: List of embedding vectors
            num_clusters: Number of clusters to form with GMM.
            verbose: If True, prints detailed cluster information.

        Returns:
            List of cluster assignments for each embedding.
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(embeddings)
        clusters = gmm.predict(embeddings)

        if verbose:
            logger.info(
                f"GMM {num_clusters} Clusters ==> Got {len(embeddings)} "
                f"embeddings with cluster assignments:"
            )
            num_members = pd.Series(clusters).value_counts()
            logger.info(num_members)

        return clusters
