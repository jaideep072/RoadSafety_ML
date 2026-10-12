"""
K-Means Clustering implementation for RoadSafety_ML.
Implements feature scaling, cluster label assignment, inertia computation,
elbow method optimization, and cluster visualization.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import CHARTS_DIR
from app.data.validators import validate_clustering_data
from app.evaluation.clustering import evaluate_clustering_model


def run_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 4,
    compute_elbow: bool = True,
    sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes K-Means accident clustering:
    1. Validates and scales numeric features.
    2. Fits KMeans(n_clusters).
    3. Calculates cluster metrics and cluster distributions.
    4. Computes Elbow curve across k in [2..8].
    5. Generates 2D cluster scatter plot.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = ["Start_Lng", "Start_Lat", "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)"]

    # Filter to features present in df
    active_features = [f for f in feature_cols if f in df.columns]
    if len(active_features) < 2:
        active_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Severity", "Severity_Severe"]][:4]

    clean_subset = validate_clustering_data(df, active_features, min_rows=20)

    # Sample if dataset is large for performance
    is_sampled = False
    if len(clean_subset) > sample_size:
        clean_subset = clean_subset.sample(n=sample_size, random_state=random_state)
        is_sampled = True

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_subset[active_features])

    # Fit target model
    n_clusters = max(2, min(n_clusters, 12, len(clean_subset) - 1))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    # Evaluate metrics
    metrics = evaluate_clustering_model(
        X=X_scaled,
        labels=labels,
        inertia=float(kmeans.inertia_),
        max_silhouette_samples=5000
    )

    # Cluster distribution stats
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_stats = []
    total_pts = len(labels)
    for l_id, cnt in zip(unique_labels, counts):
        cluster_stats.append({
            "cluster_id": int(l_id),
            "name": f"Cluster {l_id + 1}",
            "count": int(cnt),
            "percentage": round(cnt / total_pts * 100.0, 2)
        })

    # Cluster centroids in original feature scale
    original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_summary = []
    for idx, center in enumerate(original_centers):
        center_dict = {"cluster": f"Cluster {idx + 1}"}
        for f_name, f_val in zip(active_features, center):
            center_dict[f_name] = round(float(f_val), 2)
        centers_summary.append(center_dict)

    # 1. Cluster Scatter Visualization
    plot_x_col = active_features[0]
    plot_y_col = active_features[1] if len(active_features) > 1 else active_features[0]

    scatter_filename = f"kmeans_clusters_k{n_clusters}.png"
    plt.figure(figsize=(7, 5), dpi=120)
    palette = sns.color_palette("tab10", n_clusters)
    scatter = sns.scatterplot(
        x=clean_subset[plot_x_col],
        y=clean_subset[plot_y_col],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=30,
        legend="full"
    )
    plt.title(f"K-Means Accident Clustering (k={n_clusters})", fontsize=11, fontweight="bold")
    plt.xlabel(plot_x_col, fontsize=10)
    plt.ylabel(plot_y_col, fontsize=10)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / scatter_filename, bbox_inches="tight")
    plt.close()

    # 2. Elbow Method Curve
    elbow_filename = "kmeans_elbow_curve.png"
    elbow_data = []
    if compute_elbow:
        k_range = range(2, min(9, len(clean_subset)))
        inertias = []
        for k in k_range:
            km_k = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=random_state)
            km_k.fit(X_scaled)
            inertias.append(km_k.inertia_)
            elbow_data.append({"k": k, "inertia": round(float(km_k.inertia_), 2)})

        plt.figure(figsize=(6.5, 4), dpi=120)
        plt.plot(list(k_range), inertias, marker="o", color="#0284c7", linewidth=2, markersize=6)
        plt.axvline(x=n_clusters, color="#ef4444", linestyle="--", linewidth=1.5,
                    label=f"Selected k = {n_clusters}")
        plt.title("K-Means Elbow Method (Inertia vs. Number of Clusters)", fontsize=11, fontweight="bold")
        plt.xlabel("Number of Clusters (k)", fontsize=10)
        plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / elbow_filename, bbox_inches="tight")
        plt.close()

    return {
        "algorithm": "K-Means",
        "n_clusters": n_clusters,
        "features": active_features,
        "total_samples": total_pts,
        "is_sampled": is_sampled,
        "metrics": metrics.to_dict(),
        "cluster_stats": cluster_stats,
        "centers_summary": centers_summary,
        "elbow_data": elbow_data,
        "scatter_plot": scatter_filename,
        "elbow_plot": elbow_filename if compute_elbow else None
    }
