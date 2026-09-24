"""
CLI script to execute Unsupervised Clustering (K-Means, DBSCAN with Knee, Hierarchical).
Usage: python scripts/run_clustering.py [--algo kmeans|dbscan|hierarchical|geographic]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.loader import load_dataset
from app.clustering.kmeans import run_kmeans_clustering
from app.clustering.dbscan import run_dbscan_clustering
from app.clustering.knee_for_dbscan import find_knee_for_dbscan
from app.clustering.hierarchical import run_hierarchical_clustering
from app.clustering.geographic import run_geographic_analysis


def main():
    parser = argparse.ArgumentParser(description="Run RoadSafety_ML Unsupervised Clustering")
    parser.add_argument("--algo", choices=["kmeans", "dbscan", "hierarchical", "geographic", "all"], default="all")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters for K-Means/Hierarchical")
    parser.add_argument("--eps", type=float, default=None, help="Epsilon for DBSCAN (None = Auto-Knee)")
    parser.add_argument("--min-samples", type=int, default=5, help="Min samples for DBSCAN")
    args = parser.parse_args()

    print("==================================================")
    print("  RoadSafety_ML — Unsupervised Clustering Suite")
    print("==================================================")

    df = load_dataset(dataset_type="preprocessed")

    if args.algo in ["kmeans", "all"]:
        print("\n--- 1. Running K-Means Clustering ---")
        km_res = run_kmeans_clustering(df, n_clusters=args.k)
        print(f"  Clusters: {km_res['n_clusters']}")
        print(f"  Inertia : {km_res['metrics']['inertia']}")
        print(f"  Silhouette Score: {km_res['metrics']['silhouette']}")
        print(f"  Davies-Bouldin  : {km_res['metrics']['davies_bouldin']}")

    if args.algo in ["dbscan", "all"]:
        print("\n--- 2. Running DBSCAN Clustering (Auto-Knee) ---")
        db_res = run_dbscan_clustering(df, eps=args.eps, min_samples=args.min_samples, auto_eps=(args.eps is None))
        print(f"  Epsilon (eps): {db_res['eps']}")
        print(f"  Clusters Found: {db_res['n_clusters']}")
        print(f"  Core Points: {db_res['n_core']}")
        print(f"  Border Points: {db_res['n_border']}")
        print(f"  Noise Outliers: {db_res['n_noise']}")
        print(f"  Silhouette Score: {db_res['metrics']['silhouette']}")

    if args.algo in ["hierarchical", "all"]:
        print("\n--- 3. Running Hierarchical Agglomerative Clustering ---")
        hi_res = run_hierarchical_clustering(df, n_clusters=args.k, linkage_type="ward")
        print(f"  Clusters: {hi_res['n_clusters']}")
        print(f"  Linkage: {hi_res['linkage']}")
        print(f"  Silhouette Score: {hi_res['metrics']['silhouette']}")

    if args.algo in ["geographic", "all"]:
        print("\n--- 4. Running Geographic Spatial Analysis ---")
        geo_res = run_geographic_analysis(df)
        print(f"  Total Geo Records: {geo_res['total_points']}")
        print(f"  Center Coordinates: {geo_res['center']}")

    print("\n==================================================")
    print("  Clustering execution completed!")
    print("==================================================")


if __name__ == "__main__":
    main()
