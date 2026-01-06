import numpy as np
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def get_dim_reduction(
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


def plot_reduced_embeddings(
    embeddings: list,
    method: str = "pca",
    num_dims: int = 3,
    title="Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    """
    Plot embeddings in reduced dimensionality.

    Args:
        embeddings: List of embedding vectors
        method: Dimensionality reduction method
        num_dims: Number of dimensions (2 or 3)
        title: Plot title
        cluster_ids: Optional cluster assignments for coloring
        cluster_label: Label for the cluster colorbar
        patch_type: Optional patch types for different markers

    Returns:
        Figure and axes objects
    """
    transformed = get_dim_reduction(embeddings, method, num_dims)

    if num_dims == 2:
        fig, ax = plot_2d_scatter(
            transformed, title, cluster_ids, cluster_label, patch_type
        )
    elif num_dims == 3:
        fig, ax = plot_3d_scatter(
            transformed, title, cluster_ids, cluster_label, patch_type
        )
    else:
        raise ValueError(f"Invalid number of dimensions: {num_dims}")

    return fig, ax


def plot_2d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    """Plot 2D scatter of embeddings."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 7))

    if cluster_ids is not None:
        _, cluster_ids_for_coloring = np.unique(cluster_ids, return_inverse=True)
        num_distinct_colors = len(np.unique(cluster_ids))
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        num_distinct_colors = 1

    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
    colors_for_cmap = (base_colors * multiplier)[:num_distinct_colors]
    cmap = ListedColormap(colors_for_cmap)
    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            scatter_args = {
                "marker": marker_shapes[i % len(marker_shapes)],
                "alpha": 0.6,
                "s": 100,
                "label": str(patch_val),
            }
            if cluster_ids is not None:
                scatter_args["c"] = cluster_ids_for_coloring[patch_mask]
                scatter_args["cmap"] = cmap
            ax.scatter(
                transformed[patch_mask, 0], transformed[patch_mask, 1], **scatter_args
            )
    else:
        scatter_args = {"marker": "o", "alpha": 0.6, "s": 100}
        if cluster_ids is not None:
            scatter_args["c"] = cluster_ids_for_coloring
            scatter_args["cmap"] = cmap
        ax.scatter(transformed[:, 0], transformed[:, 1], **scatter_args)

    ax.set_xlabel("1st Latent Dim.", fontsize=20)
    ax.set_ylabel("2nd Latent Dim.", fontsize=20)
    ax.set_title(title, fontsize=30)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if patch_type is not None:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_shapes[i % len(marker_shapes)],
                color="black",
                label=str(pv),
                linestyle="None",
                markersize=10,
            )
            for i, pv in enumerate(np.unique(np.array(patch_type)))
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, title="Patch Types", loc="best")

    fig.tight_layout()
    return fig, ax


def plot_3d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    """Plot 3D scatter of embeddings."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    if cluster_ids is not None:
        _, cluster_ids_for_coloring = np.unique(cluster_ids, return_inverse=True)
        num_distinct_colors = len(np.unique(cluster_ids))
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        num_distinct_colors = 1

    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
    colors_for_cmap = (base_colors * multiplier)[:num_distinct_colors]
    cmap = ListedColormap(colors_for_cmap)
    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            scatter_args = {
                "marker": marker_shapes[i % len(marker_shapes)],
                "alpha": 0.6,
                "s": 20,
                "label": str(patch_val),
            }
            if cluster_ids is not None:
                scatter_args["c"] = cluster_ids_for_coloring[patch_mask]
                scatter_args["cmap"] = cmap
            ax.scatter(
                transformed[patch_mask, 0],
                transformed[patch_mask, 1],
                transformed[patch_mask, 2],
                **scatter_args,
            )
    else:
        scatter_args = {"marker": "o", "alpha": 0.6, "s": 20}
        if cluster_ids is not None:
            scatter_args["c"] = cluster_ids_for_coloring
            scatter_args["cmap"] = cmap
        ax.scatter(
            transformed[:, 0], transformed[:, 1], transformed[:, 2], **scatter_args
        )

    ax.set_xlabel("1st Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_ylabel("2nd Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_zlabel("3rd Latent Dim.", labelpad=-17, rotation=90, fontsize=8)
    ax.set_title(title, y=0.95)

    if patch_type is not None:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_shapes[i % len(marker_shapes)],
                color="black",
                label=str(pv),
                linestyle="None",
                markersize=10,
            )
            for i, pv in enumerate(np.unique(np.array(patch_type)))
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title="Patch Types",
                loc="best",
                bbox_to_anchor=(0.9, 0.5),
            )

    ax.view_init(elev=20, azim=45)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)
    fig.tight_layout()
    return fig, ax
