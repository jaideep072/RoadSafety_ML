"""
Principal Component Analysis (PCA) Module for RoadSafety_ML.
Implements unsupervised dimensionality reduction, scree plot generation,
cumulative explained variance analysis, and 2D principal component projections
strictly aligned with Module 4 Session 4 (PCA) curriculum.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.config import (
    CHARTS_DIR,
    STATIC_DIR,
    DEFAULT_NUMERIC_FEATURES
)
from app.data.loader import load_dataset


def run_pca_analysis(n_components: int = 5, sample_size: int = 5000) -> Dict[str, Any]:
    """
    Executes PCA on continuous environmental features:
    1. Standardizes features with StandardScaler (essential for PCA).
    2. Fits PCA to compute eigenvalues, eigenvectors (components), and explained variance ratio.
    3. Generates Scree Plot with Elbow marker and Cumulative Variance step-curve.
    4. Generates 2D PC1 vs PC2 projection scatter plot colored by accident severity.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_type="preprocessed")

    # Features strictly continuous as per M4-Session 4:
    features = [f for f in DEFAULT_NUMERIC_FEATURES if f in df.columns]
    if len(features) < 2:
        features = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]

    # Sample for crisp, responsive plotting
    sample_n = min(sample_size, len(df))
    df_sample = df.sample(n=sample_n, random_state=42).copy()

    X = df_sample[features]
    y_severity = df_sample["Severity"].values if "Severity" in df_sample.columns else np.ones(sample_n)

    # 1. StandardScaler (Mean = 0, Variance = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit Full PCA
    max_comps = min(n_components, len(features))
    pca = PCA(n_components=max_comps, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Variance Metrics
    var_ratio = [float(v) for v in pca.explained_variance_ratio_]
    cum_var_ratio = [float(v) for v in np.cumsum(var_ratio)]
    eigenvalues = [float(v) for v in pca.explained_variance_]

    # Scree Plot (Explained Variance & Cumulative)
    scree_plot_filename = "pca_scree_plot.png"
    plt.figure(figsize=(8, 4.5), dpi=130)
    comps = [f"PC{i+1}" for i in range(max_comps)]
    
    # Bar chart for individual variance
    bars = plt.bar(comps, [v * 100 for v in var_ratio], color="#0284c7", alpha=0.8, label="Individual Explained Variance (%)")
    # Step/Line for cumulative variance
    plt.plot(comps, [v * 100 for v in cum_var_ratio], color="#ef4444", marker="o", linewidth=2.2, label="Cumulative Explained Variance (%)")
    plt.axhline(y=80, color="#10b981", linestyle="--", alpha=0.7, label="80% Information Threshold")

    for bar, pct in zip(bars, var_ratio):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{pct*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.xlabel("Principal Components", fontsize=10, fontweight="bold")
    plt.ylabel("Variance Explained (%)", fontsize=10, fontweight="bold")
    plt.title("PCA Scree Plot & Cumulative Information Retained", fontsize=11, fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(loc="center right", fontsize=8.5)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / scree_plot_filename)
    plt.savefig(CHARTS_DIR / scree_plot_filename)
    plt.close()

    # 2D PC1 vs PC2 Scatter Projection
    scatter_plot_filename = "pca_projection_scatter.png"
    plt.figure(figsize=(7.5, 5), dpi=130)
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_severity,
        cmap="coolwarm",
        alpha=0.6,
        s=22,
        edgecolors="none"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Accident Severity Level (1 - 4)", fontsize=9, fontweight="bold")
    plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    plt.title("2D Principal Component Projection (PC1 vs PC2)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / scatter_plot_filename)
    plt.savefig(CHARTS_DIR / scatter_plot_filename)
    plt.close()

    # Eigenvectors (Component Loadings Table)
    loadings = []
    for i in range(max_comps):
        row = {
            "pc": f"PC{i+1}",
            "variance_pct": round(var_ratio[i] * 100, 2),
            "cum_variance_pct": round(cum_var_ratio[i] * 100, 2),
            "eigenvalue": round(eigenvalues[i], 4),
            "weights": {feat: round(float(pca.components_[i, idx]), 4) for idx, feat in enumerate(features)}
        }
        loadings.append(row)

    return {
        "features": features,
        "n_features": len(features),
        "n_samples": sample_n,
        "scree_plot": scree_plot_filename,
        "scatter_plot": scatter_plot_filename,
        "var_ratio": [round(v * 100, 2) for v in var_ratio],
        "cum_var_ratio": [round(v * 100, 2) for v in cum_var_ratio],
        "top2_var_pct": round(cum_var_ratio[1] * 100, 1),
        "loadings": loadings
    }
