"""
Decision Tree and Ensemble Models trainer and evaluator for RoadSafety_ML.
Computes non-linear accident severity classification, feature importances,
confusion matrices, tree depth visualization, and real-time live inference.
Includes comprehensive 7-ensemble benchmark leaderboard with champion architecture tracking.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import joblib

from app.config import (
    CHARTS_DIR,
    STATIC_DIR,
    MODELS_DIR,
    SUPERVISED_CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET
)
from app.data.loader import load_dataset
from app.evaluation.classification import evaluate_classification_model
from app.models.ensembles import ALGO_CONFIG, train_ensemble_model
from app.models.diagnostic_curves import generate_diagnostic_curves

BENCHMARK_CACHE_PATH = MODELS_DIR / "tree_benchmark_cache.pkl"


def get_all_trees_benchmark(X_train, X_test, y_train, y_test, active_features):
    """
    Computes or retrieves cached benchmark results across all 7 ensemble algorithms.
    """
    if BENCHMARK_CACHE_PATH.exists():
        try:
            return joblib.load(BENCHMARK_CACHE_PATH)
        except Exception:
            pass

    benchmark_list = []
    models_summary = {}

    for k, cfg in ALGO_CONFIG.items():
        t0 = time.time()
        mod = cfg["classifier"]()
        mod.fit(X_train, y_train)
        training_time = round(time.time() - t0, 2)

        y_tr_pred = mod.predict(X_train)
        y_te_pred = mod.predict(X_test)
        
        y_te_prob = None
        if hasattr(mod, "predict_proba"):
            try:
                y_te_prob = mod.predict_proba(X_test)[:, 1]
            except Exception:
                y_te_prob = None

        tr_ev = evaluate_classification_model(y_train, y_tr_pred)
        te_ev = evaluate_classification_model(y_test, y_te_pred, y_prob=y_te_prob)

        item = {
            "id": k,
            "name": cfg["name"],
            "dropdown_name": cfg["name"],
            "family": cfg["category"],
            "description": cfg["description"],
            "train_accuracy": tr_ev.accuracy,
            "test_accuracy": te_ev.accuracy,
            "precision": te_ev.precision,
            "recall": te_ev.recall,
            "f1": te_ev.f1,
            "roc_auc": round(float(te_ev.roc_auc), 4) if te_ev.roc_auc else None,
            "tp": te_ev.tp,
            "tn": te_ev.tn,
            "fp": te_ev.fp,
            "fn": te_ev.fn,
            "training_time": training_time,
            "overfitting_gap": round(float(tr_ev.accuracy - te_ev.accuracy), 2)
        }
        benchmark_list.append(item)
        models_summary[k] = item

    # Sort by test accuracy descending to find champion
    benchmark_sorted = sorted(benchmark_list, key=lambda x: x["test_accuracy"], reverse=True)
    champion = benchmark_sorted[0]

    cache_data = {
        "benchmark": benchmark_sorted,
        "models_summary": models_summary,
        "champion": champion
    }

    try:
        joblib.dump(cache_data, BENCHMARK_CACHE_PATH)
    except Exception:
        pass

    return cache_data


def train_and_evaluate_tree(
    algo_key: str = "id3",
    student_input: Optional[Dict[str, Any]] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Trains selected Decision Tree or Ensemble algorithm, calculates metrics,
    plots confusion matrix and feature importances, and infers user scenario.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if algo_key not in ALGO_CONFIG:
        algo_key = "id3"

    config = ALGO_CONFIG[algo_key]
    model = config["classifier"]()

    df = load_dataset(dataset_type="preprocessed")

    active_features = [f for f in SUPERVISED_CLASSIFICATION_FEATURES if f in df.columns]
    target_col = CLASSIFICATION_TARGET if CLASSIFICATION_TARGET in df.columns else "Severity"

    if target_col == "Severity" and df["Severity"].nunique() > 2:
        y = (df["Severity"] >= 3).astype(int)
    else:
        y = df[target_col].astype(int)

    X = df[active_features].copy()

    # 3-Way Disjoint Partition (Train ~70%, Validation ~15%, Test ~15%) - Module 5 Session M5-1
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = round(time.time() - t0, 2)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_test_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_test_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_test_prob = None

    train_metrics = evaluate_classification_model(y_train, y_train_pred)
    val_metrics = evaluate_classification_model(y_val, y_val_pred)
    test_metrics = evaluate_classification_model(y_test, y_test_pred, y_prob=y_test_prob)

    # Feature Importances Extraction
    feature_importances: List[Dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        for f, imp in zip(active_features, raw_imp):
            feature_importances.append({
                "feature": f,
                "importance": round(float(imp) * 100, 2)
            })
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        mean_imp = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        for f, imp in zip(active_features, mean_imp):
            feature_importances.append({
                "feature": f,
                "importance": round(float(imp) * 100, 2)
            })
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

    # Add 1-based ranks to feature importances
    for idx, item in enumerate(feature_importances):
        item["rank"] = idx + 1

    # Visualization: Confusion Matrix
    cm_plot = f"dt_cm_{algo_key}.png"
    cm = np.array(test_metrics.confusion_matrix)

    plt.figure(figsize=(6, 4.5), dpi=130)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=["Minor / Moderate", "Severe"],
        yticklabels=["Minor / Moderate", "Severe"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.xlabel("Predicted Severity Class", fontsize=10, fontweight="bold", color="#1e293b", labelpad=8)
    plt.ylabel("Actual Severity Class", fontsize=10, fontweight="bold", color="#1e293b", labelpad=8)
    plt.title(f"Confusion Matrix — {config['name']}", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / cm_plot)
    plt.savefig(CHARTS_DIR / cm_plot)
    plt.close("all")

    # Visualization: Feature Importances Bar Chart
    imp_plot = f"dt_imp_{algo_key}.png"
    if feature_importances:
        top_imp = feature_importances[:8]
        plt.figure(figsize=(7.5, 4.5), dpi=130)
        names = [item["feature"] for item in reversed(top_imp)]
        vals = [item["importance"] for item in reversed(top_imp)]
        plt.barh(names, vals, color="#2563eb", height=0.55, edgecolor="none")
        plt.xlabel("Gini / Split Feature Importance Share (%)", fontsize=9.5, color="#334155", labelpad=6)
        plt.title(f"Top Predictor Features Ranking — {config['name']}", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
        plt.tight_layout()
        plt.savefig(STATIC_DIR / imp_plot)
        plt.savefig(CHARTS_DIR / imp_plot)
        plt.close("all")
    else:
        imp_plot = None

    # Visualization: Decision Tree Graph (For single Decision Tree models: CART or ID3)
    tree_plot = None
    if algo_key in ["cart", "id3", "dt"]:
        tree_plot = f"dt_tree_{algo_key}.png"
        fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
        plot_tree(
            model,
            feature_names=active_features,
            class_names=["Minor/Mod", "Severe"],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=3,
            ax=ax
        )
        plt.title(f"Level-Wise Decision Tree Architecture (Levels 0 to 3) — {config['name']}", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(STATIC_DIR / tree_plot, bbox_inches="tight", facecolor="white")
        plt.savefig(CHARTS_DIR / tree_plot, bbox_inches="tight", facecolor="white")
        plt.close("all")

    # Hyperparameters dictionary
    hyperparams = config.get("hyperparams", {"random_state": "42"})

    # Single-Instance Live Predictor
    prediction = None
    if student_input:
        prediction = predict_tree_severity(model, student_input, active_features)

    # Optional model saving
    if save_model:
        model_filename = f"{config['file_prefix']}_classifier.pkl"
        joblib.dump(model, MODELS_DIR / model_filename)

    # Get multi-model benchmark leaderboard
    benchmark_data = get_all_trees_benchmark(X_train, X_test, y_train, y_test, active_features)

    return {
        "algo_key": algo_key,
        "algo_name": config["name"],
        "category": config["category"],
        "description": config["description"],
        "training_time": training_time,
        "hyperparams": hyperparams,
        "metadata": {
            "target_variable": "Severity_Severe (1 = Level 3-4, 0 = Level 1-2)",
            "training_samples": f"{len(X_train):,} (70%)",
            "validation_samples": f"{len(X_val):,} (15%)",
            "testing_samples": f"{len(X_test):,} (15%)",
            "features_evaluated": f"{len(active_features)} features",
            "evaluation_split": "70% Train / 15% Validation / 15% Stratified Test Split (Module 5)"
        },
        "metrics": {
            "test_accuracy": test_metrics.accuracy,
            "val_accuracy": val_metrics.accuracy,
            "train_accuracy": train_metrics.accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1,
            "roc_auc": round(float(test_metrics.roc_auc), 4) if test_metrics.roc_auc else None,
            "tp": test_metrics.tp,
            "tn": test_metrics.tn,
            "fp": test_metrics.fp,
            "fn": test_metrics.fn,
            "overfitting_gap": round(float(train_metrics.accuracy - test_metrics.accuracy), 2)
        },
        "feature_importances": feature_importances,
        "plots": {
            "cm_plot": cm_plot,
            "imp_plot": imp_plot,
            "tree_plot": tree_plot
        },
        "prediction": prediction,
        "algo_list": [(k, v["name"]) for k, v in ALGO_CONFIG.items()],
        "benchmark": benchmark_data["benchmark"],
        "champion": benchmark_data["champion"],
        "features": active_features,
        "diagnostics": generate_diagnostic_curves()
    }


def predict_tree_severity(
    model,
    input_data: Dict[str, Any],
    features: List[str]
) -> Optional[Dict[str, Any]]:
    """Infers severity class and probability for tree/ensemble estimators."""
    try:
        row_vals = [float(input_data.get(f, 0.0)) for f in features]
        input_df = pd.DataFrame([row_vals], columns=features)
        pred_class = int(model.predict(input_df)[0])
        try:
            prob = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)
        except Exception:
            prob = 85.0 if pred_class == 1 else 15.0
        return {
            "class": "Severe Accident (Severity 3-4)" if pred_class == 1 else "Minor / Moderate Accident (Severity 1-2)",
            "is_severe": pred_class == 1,
            "probability": prob
        }
    except Exception:
        return None
