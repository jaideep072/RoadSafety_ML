"""
CLI script to execute the 10+ Model Training and Benchmarking Suite.
Usage: python scripts/train_models.py [--force]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.pipeline import run_models_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train RoadSafety_ML Supervised Models")
    parser.add_argument("--force", action="store_true", help="Force retraining of all models")
    args = parser.parse_args()

    print("==================================================")
    print("  RoadSafety_ML — Model Training & Benchmarking")
    print("==================================================")
    results = run_models_pipeline(force_run=args.force)

    print("\n--- Regression Model Leaderboard ---")
    print(f"{'Model Name':<32} | {'R² (%)':<8} | {'RMSE':<8} | {'MAE':<8}")
    print("-" * 62)
    for m in results.get("regression_leaderboard", []):
        print(f"{m['name']:<32} | {m['r2']:<8.2f} | {m['rmse']:<8.4f} | {m['mae']:<8.4f}")

    print("\n--- Classification Model Leaderboard ---")
    print(f"{'Model Name':<32} | {'Accuracy (%)':<14} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'ROC-AUC':<8}")
    print("-" * 88)
    for m in results.get("classification_leaderboard", []):
        auc_str = f"{m['roc_auc']:.4f}" if m.get('roc_auc') is not None else "N/A"
        print(f"{m['name']:<32} | {m['accuracy']:<14.2f} | {m['precision']:<14.2f} | {m['recall']:<12.2f} | {auc_str:<8}")

    print("==================================================")


if __name__ == "__main__":
    main()
