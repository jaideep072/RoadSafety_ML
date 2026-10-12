"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) module for RoadSafety_ML.
Identifies arbitrary-shaped accident density clusters, core points, border points, and spatial outliers.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.config import CHARTS_DIR
from app.data.validators import validate_clustering_data
from app.evaluation.clustering import evaluate_clustering_model
from app.clustering.knee_for_dbscan import find_knee_for_dbscan


def run_dbscan_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    eps: Optional[float] = None,
    min_samples: int = 5,
    auto_eps: bool = False,
    sample_size: int = 5000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes DBSCAN clustering on accident records:
    1. Selects numeric features.
    2. Determines automated eps via Knee graph if requested or eps is None.
    3. Fits DBSCAN.
    4. Identifies Core points, Border points, and Noise points (-1).
    5. Computes cluster metrics and cluster distributions.
    6. Generates cluster density scatter chart.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []

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
        warnings.append(f"DBSCAN executed on a sample of {sample_size:,} records for responsiveness.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_subset[active_features])

    # Automated eps detection via Knee method
    knee_metadata = None
    if auto_eps or eps is None:
        knee_metadata = find_knee_for_dbscan(
            df_features=clean_subset[active_features].values,
            min_samples=min_samples,
            sample_size=sample_size,
            save_plot=True
        )
        eps = float(knee_metadata["recommended_eps"])
        warnings.extend(knee_metadata.get("warnings", []))

    # Fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # Core, border, noise breakdown
    core_indices = set(db.core_sample_indices_)
    n_total = len(labels)
    n_core = len(core_indices)
    n_noise = int(np.sum(labels == -1))
    n_border = n_total - n_core - n_noise

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})

    # Cluster metrics
    metrics = evaluate_clustering_model(
        X=X_scaled,
        labels=labels,
        inertia=None,
        max_silhouette_samples=5000
    )

    # Cluster distribution stats
    cluster_stats = []
    for l_id in sorted(unique_labels):
        cnt = int(np.sum(labels == l_id))
        pct = round(cnt / n_total * 100.0, 2)
        if l_id == -1:
            name = "Noise / Outliers"
        else:
            name = f"Cluster {l_id + 1} (Concentration Zone)"
        cluster_stats.append({
            "cluster_id": int(l_id),
            "name": name,
            "count": cnt,
            "percentage": pct
        })

    # Cluster Scatter Plot
    plot_x_col = active_features[0]
    plot_y_col = active_features[1] if len(active_features) > 1 else active_features[0]
    scatter_filename = f"dbscan_clusters_eps_{str(eps).replace('.', '_')}.png"

    plt.figure(figsize=(7.5, 5), dpi=120)
    
    # Plot noise points first in gray
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        plt.scatter(
            clean_subset[plot_x_col].iloc[noise_mask],
            clean_subset[plot_y_col].iloc[noise_mask],
            c="#94a3b8",
            label="Noise / Dispersed",
            alpha=0.3,
            s=15,
            edgecolors="none"
        )

    # Plot cluster points
    cluster_mask = ~noise_mask
    if np.any(cluster_mask):
        sns.scatterplot(
            x=clean_subset[plot_x_col].iloc[cluster_mask],
            y=clean_subset[plot_y_col].iloc[cluster_mask],
            hue=labels[cluster_mask],
            palette="tab10",
            alpha=0.7,
            s=35,
            legend="full"
        )

    plt.title(f"DBSCAN Density Clustering (eps={eps:.3f}, min_samples={min_samples})", fontsize=11, fontweight="bold")
    plt.xlabel(plot_x_col, fontsize=10)
    plt.ylabel(plot_y_col, fontsize=10)
    plt.legend(title="Density Clusters", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / scatter_filename, bbox_inches="tight")
    plt.close()

    return {
        "algorithm": "DBSCAN",
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_core": n_core,
        "n_border": n_border,
        "features": active_features,
        "total_samples": n_total,
        "is_sampled": is_sampled,
        "metrics": metrics.to_dict(),
        "cluster_stats": cluster_stats,
        "knee_metadata": knee_metadata,
        "scatter_plot": scatter_filename,
        "warnings": warnings
    }
