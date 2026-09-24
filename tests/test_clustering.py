"""
Unit tests for unsupervised clustering modules: K-Means, DBSCAN,
automated Knee eps detection, Hierarchical clustering, and Geographic analysis.
"""

import pytest
import numpy as np
import pandas as pd

from app.data.loader import load_dataset
from app.clustering.kmeans import run_kmeans_clustering
from app.clustering.dbscan import run_dbscan_clustering
from app.clustering.knee_for_dbscan import find_knee_for_dbscan, calculate_k_distances, detect_knee_point
from app.clustering.hierarchical import run_hierarchical_clustering
from app.clustering.geographic import run_geographic_analysis


@pytest.fixture
def clustering_data():
    df = load_dataset(dataset_type="preprocessed", nrows=1000)
    return df


def test_kmeans_clustering(clustering_data):
    res = run_kmeans_clustering(clustering_data, n_clusters=3, compute_elbow=True, sample_size=500)
    assert res["algorithm"] == "K-Means"
    assert res["n_clusters"] == 3
    assert "metrics" in res
    assert "inertia" in res["metrics"]
    assert len(res["cluster_stats"]) == 3
    assert len(res["elbow_data"]) > 0


def test_knee_for_dbscan():
    # Synthetic sorted distances curve
    np.random.seed(42)
    X = np.random.randn(100, 3)
    k_dist = calculate_k_distances(X, k=4)
    assert len(k_dist) == 100
    assert np.all(np.diff(k_dist) >= 0)  # Verify ascending sort

    eps, knee_idx = detect_knee_point(k_dist)
    assert eps > 0.0
    assert 0 <= knee_idx < len(k_dist)


def test_dbscan_clustering(clustering_data):
    res = run_dbscan_clustering(clustering_data, eps=0.5, min_samples=5, auto_eps=False, sample_size=500)
    assert res["algorithm"] == "DBSCAN"
    assert "n_clusters" in res
    assert "n_core" in res
    assert "n_noise" in res
    assert "n_border" in res
    assert res["n_core"] + res["n_border"] + res["n_noise"] == res["total_samples"]


def test_dbscan_auto_knee(clustering_data):
    res = run_dbscan_clustering(clustering_data, auto_eps=True, min_samples=4, sample_size=500)
    assert res["algorithm"] == "DBSCAN"
    assert res["eps"] > 0.0
    assert res["knee_metadata"] is not None


def test_hierarchical_clustering(clustering_data):
    res = run_hierarchical_clustering(clustering_data, n_clusters=3, linkage_type="ward", sample_size=300)
    assert res["algorithm"] == "Hierarchical Clustering"
    assert res["n_clusters"] == 3
    assert res["linkage"] == "ward"
    assert len(res["cluster_stats"]) == 3


def test_geographic_analysis(clustering_data):
    res = run_geographic_analysis(clustering_data, sample_size=500)
    assert "total_points" in res
    assert "lat_range" in res
    assert "lng_range" in res
    assert "center" in res
