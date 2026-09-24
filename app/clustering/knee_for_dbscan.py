"""
Automated DBSCAN Epsilon (eps) Detection using the K-Distance Graph & Knee Point Method.
Calculates sorted k-nearest neighbor distances and determines the elbow/knee inflection point.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.config import CHARTS_DIR


def calculate_k_distances(
    X_scaled: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Computes sorted distances to the k-th nearest neighbor for each sample.
    """
    n_samples = len(X_scaled)
    actual_k = min(k, max(2, n_samples - 1))
    
    neigh = NearestNeighbors(n_neighbors=actual_k, metric="euclidean", n_jobs=-1)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)

    # Distances are sorted per row; take the k-th neighbor (last column)
    k_distances = distances[:, -1]
    sorted_k_distances = np.sort(k_distances)
    return sorted_k_distances


def detect_knee_point(sorted_distances: np.ndarray) -> Tuple[float, int]:
    """
    Finds the maximum perpendicular distance from the curve to the line
    connecting the first and last points (Kneedle algorithm in pure NumPy).
    """
    n = len(sorted_distances)
    if n < 3:
        return float(sorted_distances[-1]) if n > 0 else 0.5, 0

    x = np.linspace(0, 1, n)
    y = (sorted_distances - sorted_distances[0]) / (sorted_distances[-1] - sorted_distances[0] + 1e-10)

    # Line from (0, y0) to (1, y1)
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = np.sqrt(dx**2 + dy**2)
    if line_len == 0:
        return float(sorted_distances[n // 2]), n // 2

    # Vector from line start to each point
    vx = x - x[0]
    vy = y - y[0]
    
    # 2D cross product magnitude: |dx * vy - dy * vx| / line_len
    distances_to_line = np.abs(dx * vy - dy * vx) / line_len
    knee_idx = int(np.argmax(distances_to_line))

    recommended_eps = float(sorted_distances[knee_idx])
    return recommended_eps, knee_idx


def find_knee_for_dbscan(
    df_features: np.ndarray,
    min_samples: int = 5,
    sample_size: int = 5000,
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    End-to-end automatic DBSCAN epsilon selection:
    1. Scales features using StandardScaler.
    2. Computes sorted k-distance curve with k = min_samples.
    3. Detects optimal knee point.
    4. Generates visual k-distance curve chart.
    5. Returns recommended eps, warnings, and plot metadata.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []
    
    X = np.asarray(df_features)
    if len(X) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[idx]
        warnings.append(f"K-distance computed on a representative sample of {sample_size:,} records.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = max(2, min_samples)
    sorted_distances = calculate_k_distances(X_scaled, k=k)

    try:
        recommended_eps, knee_idx = detect_knee_point(sorted_distances)
        if recommended_eps <= 0.01:
            recommended_eps = 0.3
            warnings.append("Knee detected very low distance; applied fallback eps=0.3.")
    except Exception as e:
        recommended_eps = 0.5
        knee_idx = len(sorted_distances) // 2
        warnings.append(f"Knee detection encountered fallback: {e}")

    chart_filename = "knee_dbscan_kdistance.png"
    if save_plot:
        plt.figure(figsize=(7, 4.5), dpi=120)
        plt.plot(sorted_distances, color="#0284c7", linewidth=2, label=f"{k}-NN Distance Curve")
        plt.axvline(x=knee_idx, color="#ef4444", linestyle="--", linewidth=1.5,
                    label=f"Detected Knee (Index={knee_idx})")
        plt.axhline(y=recommended_eps, color="#10b981", linestyle=":", linewidth=1.5,
                    label=f"Recommended eps = {recommended_eps:.3f}")
        plt.scatter([knee_idx], [recommended_eps], color="#ef4444", s=60, zorder=5)
        plt.xlabel("Points Sorted by Distance", fontsize=10)
        plt.ylabel(f"{k}-NN Distance (Standardized)", fontsize=10)
        plt.title(f"Automated DBSCAN K-Distance Graph (k={k})", fontsize=11, fontweight="bold")
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / chart_filename)
        plt.close()

    return {
        "recommended_eps": round(recommended_eps, 3),
        "k": k,
        "knee_index": knee_idx,
        "total_points_evaluated": len(sorted_distances),
        "chart_filename": chart_filename,
        "warnings": warnings,
        "curve_preview": [round(float(v), 3) for v in sorted_distances[::max(1, len(sorted_distances)//20)]]
    }
