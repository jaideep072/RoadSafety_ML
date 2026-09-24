"""
Clustering model evaluation metrics module.
Calculates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and Inertia.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


@dataclass
class ClusteringMetrics:
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    inertia: Optional[float]
    n_clusters: int
    n_noise: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_clustering_model(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: Optional[float] = None,
    max_silhouette_samples: int = 5000
) -> ClusteringMetrics:
    """
    Computes mathematical validation scores for unsupervised clustering partitions.
    Excludes noise points (label == -1) where necessary.
    Uses reproducible random sampling for silhouette calculation on large datasets.
    """
    labels_arr = np.asarray(labels)
    unique_labels = set(labels_arr)
    has_noise = -1 in unique_labels
    valid_cluster_labels = unique_labels - {-1}
    n_clusters = len(valid_cluster_labels)
    n_noise = int(np.sum(labels_arr == -1)) if has_noise else 0

    sil_score = None
    db_score = None
    ch_score = None

    if n_clusters >= 2:
        # Filter noise points for metric calculations if present
        if has_noise:
            mask = labels_arr != -1
            X_eval = X[mask]
            labels_eval = labels_arr[mask]
        else:
            X_eval = X
            labels_eval = labels_arr

        if len(set(labels_eval)) >= 2 and len(labels_eval) > n_clusters:
            # Silhouette score with sampling for large sets
            try:
                sample_size = min(len(labels_eval), max_silhouette_samples)
                sil_score = round(float(silhouette_score(
                    X_eval, labels_eval, sample_size=sample_size, random_state=42
                )), 4)
            except Exception:
                sil_score = None

            try:
                db_score = round(float(davies_bouldin_score(X_eval, labels_eval)), 4)
            except Exception:
                db_score = None

            try:
                ch_score = round(float(calinski_harabasz_score(X_eval, labels_eval)), 2)
            except Exception:
                ch_score = None

    return ClusteringMetrics(
        silhouette=sil_score,
        davies_bouldin=db_score,
        calinski_harabasz=ch_score,
        inertia=round(float(inertia), 2) if inertia is not None else None,
        n_clusters=n_clusters,
        n_noise=n_noise
    )
