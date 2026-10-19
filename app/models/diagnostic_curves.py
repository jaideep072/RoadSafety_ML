"""
Module 5: Diagnostic Learning Curves & Validation Curves Studio
Based on Syllabus Module 5 - Sessions M5-9 to M5-11 (Diagnostic Curves, Bias-Variance Tradeoff, and Generalization Gap).
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve
import json
from sklearn.tree import DecisionTreeClassifier

from app.config import CHARTS_DIR, DATA_DIR, PROCESSED_DATA_PATH, STATIC_DIR

logger = logging.getLogger(__name__)

DIAGNOSTIC_CACHE_PATH = STATIC_DIR / "diagnostic_curves_cache.json"


def generate_diagnostic_curves(
    force_run: bool = False
) -> Dict[str, Any]:
    """
    Computes and plots:
    1. Learning Curve: S_train(N) and S_val(N) vs Training Sample Size N
       Equation: Generalization Gap Delta(N) = S_train(N) - S_val(N)
    2. Validation Curve: Capacity vs Generalization Peak depth*
       Zones: Underfitting Zone, Optimal Peak Zone, Overfitting Zone
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    learning_curve_path = CHARTS_DIR / "diagnostic_learning_curve.png"
    validation_curve_path = CHARTS_DIR / "diagnostic_validation_curve.png"

    if not force_run and DIAGNOSTIC_CACHE_PATH.exists() and learning_curve_path.exists() and validation_curve_path.exists():
        try:
            with open(DIAGNOSTIC_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Use preprocessed dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)
    features = [
        "Temperature(F)", "Humidity(%)", "Pressure(in)",
        "Visibility(mi)", "Wind_Speed(mph)",
        "Crossing", "Junction", "Traffic_Signal"
    ]
    active_features = [f for f in features if f in df.columns]

    # Sample 8,000 rows for fast and robust cross-validation
    sample_size = min(8000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    X = sample_df[active_features]
    y = sample_df["Severity_Severe"]

    # -------------------------------------------------------------
    # 1. Learning Curve (Module 5 Slide 2 & 11)
    # -------------------------------------------------------------
    train_sizes_rel = np.linspace(0.15, 1.0, 6)
    train_sizes, train_scores, val_scores = learning_curve(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X, y,
        train_sizes=train_sizes_rel,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    tr_mean = np.mean(train_scores, axis=1) * 100
    tr_std = np.std(train_scores, axis=1) * 100
    val_mean = np.mean(val_scores, axis=1) * 100
    val_std = np.std(val_scores, axis=1) * 100

    gap_start = round(float(tr_mean[0] - val_mean[0]), 2)
    gap_end = round(float(tr_mean[-1] - val_mean[-1]), 2)

    # Plot Learning Curve
    plt.figure(figsize=(7, 4.6), dpi=140)
    plt.plot(train_sizes, tr_mean, "o-", color="#2563eb", linewidth=2.4, label="Training Score S_train(N)")
    plt.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#2563eb")

    plt.plot(train_sizes, val_mean, "s-", color="#10b981", linewidth=2.4, label="Validation Score S_val(N)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#10b981")

    plt.axhline(y=val_mean[-1], color="#94a3b8", linestyle="--", alpha=0.6, label=f"Max Val Acc ({val_mean[-1]:.1f}%)")
    plt.title("Diagnostic Learning Curve: Training vs. Validation Score", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    plt.xlabel("Training Sample Size (N)", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=8)
    plt.ylabel("Accuracy Score (%)", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(learning_curve_path, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Validation Curve across max_depth (Module 5 Slide 3 & 11)
    # -------------------------------------------------------------
    param_range = [2, 3, 4, 5, 6, 8, 10, 14]
    tr_scores_vc, val_scores_vc = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X, y,
        param_name="max_depth",
        param_range=param_range,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    vc_tr_mean = np.mean(tr_scores_vc, axis=1) * 100
    vc_tr_std = np.std(tr_scores_vc, axis=1) * 100
    vc_val_mean = np.mean(val_scores_vc, axis=1) * 100
    vc_val_std = np.std(val_scores_vc, axis=1) * 100

    best_idx = int(np.argmax(vc_val_mean))
    best_depth = param_range[best_idx]
    best_val_score = round(float(vc_val_mean[best_idx]), 2)

    # Plot Validation Curve
    plt.figure(figsize=(7, 4.6), dpi=140)
    plt.plot(param_range, vc_tr_mean, "o-", color="#2563eb", linewidth=2.4, label="Training Score (Tree Depth)")
    plt.fill_between(param_range, vc_tr_mean - vc_tr_std, vc_tr_mean + vc_tr_std, alpha=0.12, color="#2563eb")

    plt.plot(param_range, vc_val_mean, "s-", color="#d97706", linewidth=2.4, label="Validation Score (Generalization)")
    plt.fill_between(param_range, vc_val_mean - vc_val_std, vc_val_mean + vc_val_std, alpha=0.15, color="#d97706")

    # Mark the Generalization Peak depth*
    plt.axvline(x=best_depth, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Optimal Capacity depth* = {best_depth}")
    plt.scatter([best_depth], [best_val_score], color="#dc2626", s=90, zorder=5)

    plt.title("Capacity Validation Curve: Tree Depth vs. Generalization Peak", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    plt.xlabel("Hyperparameter: Tree Depth (max_depth)", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=8)
    plt.ylabel("Accuracy Score (%)", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center left", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(validation_curve_path, bbox_inches="tight")
    plt.close()

    output_data = {
        "learning_curve_plot": "charts/diagnostic_learning_curve.png",
        "validation_curve_plot": "charts/diagnostic_validation_curve.png",
        "sample_size_evaluated": sample_size,
        "gap_initial": gap_start,
        "gap_final": gap_end,
        "gap_convergence": f"{gap_start}% down to {gap_end}% (Generalization gap Delta shrinks as N increases)",
        "optimal_depth": best_depth,
        "optimal_val_accuracy": best_val_score,
        "underfitting_depth": 2,
        "overfitting_depth": 14,
        "learning_curve_table": [
            {
                "n_samples": int(n),
                "train_acc": round(float(tr), 2),
                "val_acc": round(float(va), 2),
                "gap": round(float(tr - va), 2)
            }
            for n, tr, va in zip(train_sizes, tr_mean, val_mean)
        ],
        "validation_curve_table": [
            {
                "depth": int(d),
                "train_acc": round(float(tr), 2),
                "val_acc": round(float(va), 2),
                "is_peak": (d == best_depth),
                "zone": "Optimal Peak (Good Fit)" if d == best_depth else ("Underfitting Zone" if d < best_depth else "Overfitting Zone")
            }
            for d, tr, va in zip(param_range, vc_tr_mean, vc_val_mean)
        ]
    }

    try:
        DIAGNOSTIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DIAGNOSTIC_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
    except Exception as e:
        logger.warning(f"Could not save diagnostic cache: {e}")

    return output_data
