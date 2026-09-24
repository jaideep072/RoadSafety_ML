"""
Hierarchical (Agglomerative) Clustering module for RoadSafety_ML.
Implements bottom-up hierarchical clustering, dendrogram tree generation,
linkage strategy comparisons (Ward, Complete, Average, Single), and metric evaluation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from app.config import CHARTS_DIR
from app.data.validators import validate_clustering_data
from app.evaluation.clustering import evaluate_clustering_model


def run_hierarchical_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 3,
    linkage_type: str = "ward",
    sample_size: int = 3000,
    dendrogram_sample_size: int = 150,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Agglomerative Hierarchical Clustering:
    1. Validates and scales features.
    2. Uses representative sample for model training.
    3. Fits AgglomerativeClustering(n_clusters, linkage=linkage_type).
    4. Generates Dendrogram visualization using a dedicated small reproducible sample.
    5. Generates 2D cluster scatter plot.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []

    valid_linkages = ["ward", "complete", "average", "single"]
    if linkage_type not in valid_linkages:
        linkage_type = "ward"

    if feature_cols is None:
        feature_cols = ["Start_Lng", "Start_Lat", "Temperature(F)", "Humidity(%)", "Pressure(in)"]

    active_features = [f for f in feature_cols if f in df.columns]
    if len(active_features) < 2:
        active_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Severity", "Severity_Severe"]][:4]

    clean_subset = validate_clustering_data(df, active_features, min_rows=20)

    is_sampled = False
    if len(clean_subset) > sample_size:
        clean_subset = clean_subset.sample(n=sample_size, random_state=random_state)
        is_sampled = True
        warnings.append(f"Hierarchical clustering fitted on a representative sample of {sample_size:,} records.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_subset[active_features])

    n_clusters = max(2, min(n_clusters, 8, len(clean_subset) - 1))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    labels = model.fit_predict(X_scaled)

    # Metrics
    metrics = evaluate_clustering_model(
        X=X_scaled,
        labels=labels,
        inertia=None,
        max_silhouette_samples=3000
    )

    # Cluster distribution stats
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_stats = []
    total_pts = len(labels)
    for l_id, cnt in zip(unique_labels, counts):
        cluster_stats.append({
            "cluster_id": int(l_id),
            "name": f"Hierarchical Cluster {l_id + 1}",
            "count": int(cnt),
            "percentage": round(cnt / total_pts * 100.0, 2)
        })

    # 1. Dendrogram Generation (Reproducible Sample)
    dendro_sample_n = min(dendrogram_sample_size, len(X_scaled))
    np.random.seed(random_state)
    dendro_idx = np.random.choice(len(X_scaled), size=dendro_sample_n, replace=False)
    X_dendro = X_scaled[dendro_idx]

    # Compute linkage matrix
    linkage_matrix = linkage(X_dendro, method=linkage_type)

    dendrogram_filename = f"hierarchical_dendrogram_{linkage_type}.png"
    plt.figure(figsize=(9, 4.5), dpi=120)
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )
    plt.title(
        f"Hierarchical Clustering Dendrogram (Linkage: {linkage_type.capitalize()}, Sample: {dendro_sample_n} points)",
        fontsize=11, fontweight="bold"
    )
    plt.xlabel("Sample Cluster Subtrees (p=30 condensed branches)", fontsize=10)
    plt.ylabel("Euclidean Distance / Merge Height", fontsize=10)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / dendrogram_filename, bbox_inches="tight")
    plt.close()

    # 2. Cluster Scatter Plot
    plot_x_col = active_features[0]
    plot_y_col = active_features[1] if len(active_features) > 1 else active_features[0]
    scatter_filename = f"hierarchical_clusters_k{n_clusters}_{linkage_type}.png"

    plt.figure(figsize=(7, 5), dpi=120)
    palette = sns.color_palette("tab10", n_clusters)
    sns.scatterplot(
        x=clean_subset[plot_x_col],
        y=clean_subset[plot_y_col],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=30,
        legend="full"
    )
    plt.title(f"Agglomerative Clustering (k={n_clusters}, Linkage={linkage_type.capitalize()})", fontsize=11, fontweight="bold")
    plt.xlabel(plot_x_col, fontsize=10)
    plt.ylabel(plot_y_col, fontsize=10)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / scatter_filename, bbox_inches="tight")
    plt.close()

    return {
        "algorithm": "Hierarchical Clustering",
        "n_clusters": n_clusters,
        "linkage": linkage_type,
        "features": active_features,
        "total_samples": total_pts,
        "is_sampled": is_sampled,
        "dendrogram_sample_size": dendro_sample_n,
        "metrics": metrics.to_dict(),
        "cluster_stats": cluster_stats,
        "dendrogram_plot": dendrogram_filename,
        "scatter_plot": scatter_filename,
        "warnings": warnings
    }
