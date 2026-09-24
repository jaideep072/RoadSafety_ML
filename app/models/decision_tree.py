"""
Decision Tree and Ensemble Models trainer and evaluator for RoadSafety_ML.
Computes non-linear accident severity classification, feature importances,
confusion matrices, tree depth visualization, and real-time live inference.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
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


def train_and_evaluate_tree(
    algo_key: str = "dt",
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
        algo_key = "dt"

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_test_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_test_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_test_prob = None

    train_metrics = evaluate_classification_model(y_train, y_train_pred)
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

    # Visualization: Confusion Matrix
    cm_plot = f"dt_cm_{algo_key}.png"
    cm = np.array(test_metrics.confusion_matrix)

    plt.figure(figsize=(5, 4), dpi=130)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Moderate/Minor", "Severe"],
        yticklabels=["Moderate/Minor", "Severe"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.xlabel("Predicted Severity", fontsize=10, fontweight="bold")
    plt.ylabel("Actual Severity", fontsize=10, fontweight="bold")
    plt.title(f"Confusion Matrix ({config['name']})", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / cm_plot)
    plt.savefig(CHARTS_DIR / cm_plot)
    plt.close()

    # Visualization: Feature Importances Bar Chart
    imp_plot = f"dt_imp_{algo_key}.png"
    if feature_importances:
        top_imp = feature_importances[:8]
        plt.figure(figsize=(6, 4), dpi=120)
        names = [item["feature"] for item in reversed(top_imp)]
        vals = [item["importance"] for item in reversed(top_imp)]
        plt.barh(names, vals, color="#0284c7")
        plt.xlabel("Importance (%)", fontsize=10)
        plt.title(f"Feature Importances ({config['name']})", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(STATIC_DIR / imp_plot)
        plt.savefig(CHARTS_DIR / imp_plot)
        plt.close()
    else:
        imp_plot = None

    # Visualization: Decision Tree Graph (Only for single Base Decision Tree)
    tree_plot = None
    if algo_key == "dt":
        tree_plot = "dt_tree_dt.png"
        fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
        plot_tree(
            model,
            feature_names=active_features,
            class_names=["Moderate/Minor", "Severe"],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=3,
            ax=ax
        )
        plt.title("Decision Tree Architecture (Levels 0 to 3) — Base Tree", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(STATIC_DIR / tree_plot, bbox_inches="tight", facecolor="white")
        plt.savefig(CHARTS_DIR / tree_plot, bbox_inches="tight", facecolor="white")
        plt.close()

    # Single-Instance Live Predictor
    prediction = None
    if student_input:
        prediction = predict_tree_severity(model, student_input, active_features)

    # Optional model saving
    if save_model:
        model_filename = f"{config['file_prefix']}_classifier.pkl"
        joblib.dump(model, MODELS_DIR / model_filename)

    return {
        "algo_key": algo_key,
        "algo_name": config["name"],
        "category": config["category"],
        "description": config["description"],
        "metrics": {
            "test_accuracy": test_metrics.accuracy,
            "train_accuracy": train_metrics.accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1,
            "roc_auc": test_metrics.roc_auc,
            "tp": test_metrics.tp,
            "tn": test_metrics.tn,
            "fp": test_metrics.fp,
            "fn": test_metrics.fn
        },
        "feature_importances": feature_importances,
        "plots": {
            "cm_plot": cm_plot,
            "imp_plot": imp_plot,
            "tree_plot": tree_plot
        },
        "prediction": prediction,
        "algo_list": [(k, v["name"]) for k, v in ALGO_CONFIG.items()],
        "features": active_features
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
            "class": "Severe Accident" if pred_class == 1 else "Minor / Moderate Accident",
            "is_severe": pred_class == 1,
            "probability": prob
        }
    except Exception:
        return None
