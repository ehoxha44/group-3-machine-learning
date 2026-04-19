"""
unsupervised.py – Step A: Unsupervised learning on growth trajectories.

Two algorithms
--------------
1. K-Means Clustering
   Applied to the growth-trajectory matrix (municipality × year or sector × year).
   Each row is the vector of annual YoY growth rates for that group.
   K-Means partitions municipalities / sectors into clusters with similar
   multi-year growth patterns — answering "which cities grew together?"
   and "which sectors behaved similarly over the period 2020-2025?"

2. Principal Component Analysis (PCA)
   Applied to the same feature matrix used for supervised learning.
   Reduces the dimensionality of (year, municipality_enc, sector_enc,
   prev_turnover_log1p, growth_rate, …) to 2D for scatter visualisation.
   Also used on the trajectory matrix to reveal the dominant axes of
   variation in growth patterns across groups.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    KMEANS_K,
    N_CLUSTERS_RANGE,
    N_PCA_COMPONENTS,
    RANDOM_STATE,
)


# ── K-Means on the full feature matrix ───────────────────────────────────────


def find_optimal_k(X: np.ndarray, k_range: range | None = None) -> dict:
    """Sweep K-Means over k values and record quality metrics."""
    if k_range is None:
        k_range = N_CLUSTERS_RANGE

    results: dict[str, list] = {
        "k": [], "inertia": [], "silhouette": [], "davies_bouldin": []
    }
    print("K-Means sweep:")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        n_sil = min(5_000, len(X))
        sil = silhouette_score(X, labels, sample_size=n_sil, random_state=RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(sil)
        results["davies_bouldin"].append(db)
        print(
            f"  k={k:2d}  inertia={km.inertia_:,.0f}  "
            f"silhouette={sil:.4f}  davies_bouldin={db:.4f}"
        )
    return results


def train_kmeans(X: np.ndarray, k: int = KMEANS_K) -> KMeans:
    """Fit final K-Means model."""
    print(f"Training K-Means (k={k}) …")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    km.fit(X)
    print(f"  → Inertia: {km.inertia_:,.2f}")
    return km


# ── Growth trajectory clustering ──────────────────────────────────────────────


def cluster_growth_trajectories(
    pivot: pd.DataFrame,
    k: int = KMEANS_K,
) -> tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Cluster municipalities (or sectors) by their multi-year growth trajectories.

    Parameters
    ----------
    pivot : DataFrame where rows = group (municipality/sector) and
            columns = years; values = mean YoY growth rate.
    k     : number of clusters.

    Returns
    -------
    km         : fitted KMeans model
    labels     : cluster label per row of pivot
    pivot_scaled : the StandardScaler-normalised matrix used for clustering
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot.values)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    print(
        f"  Trajectory clustering (k={k}): "
        f"silhouette={sil:.4f}  davies_bouldin={db:.4f}"
    )
    return km, labels, pd.DataFrame(X, index=pivot.index, columns=pivot.columns)


# ── PCA ───────────────────────────────────────────────────────────────────────


def apply_pca(
    X: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, PCA]:
    """Project X into n_components PCA dimensions."""
    if n_components is None:
        n_components = N_PCA_COMPONENTS
    print(f"Applying PCA (n_components={n_components}) …")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    ratios = pca.explained_variance_ratio_
    print(
        "  → Explained: " + "  ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(ratios))
        + f"  cumulative={ratios.sum():.3f}"
    )
    return X_pca, pca


def apply_pca_full(X: np.ndarray) -> PCA:
    """Fit PCA retaining all components for the variance-explained curve."""
    print("Fitting full PCA for variance analysis …")
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"  → {n95} components needed to explain ≥ 95 % variance")
    return pca
