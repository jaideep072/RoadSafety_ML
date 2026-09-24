"""
Logistic Regression models module for RoadSafety_ML.
Implements binary accident severity classification (Severe vs. Moderate/Minor)
with Unregularized (None), Ridge (L2), Lasso (L1), and ElasticNet penalties.
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
from sklearn.linear_model import LogisticRegression
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


def get_logistic_model(reg_type: str = "none"):
    """Returns initialized LogisticRegression instance."""
    reg_type = reg_type.lower()
    if reg_type == "ridge":
        return (
            LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42),
            "Logistic Regression with Ridge (L2 Penalty)"
        )
    elif reg_type == "lasso":
        return (
            LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42),
            "Logistic Regression with Lasso (L1 Penalty)"
        )
    elif reg_type == "elasticnet":
        return (
            LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=1000, random_state=42),
            "Logistic Regression with ElasticNet (L1 + L2 Penalty)"
        )
    else:
        return (
            LogisticRegression(penalty=None, max_iter=1000, random_state=42),
            "Standard Logistic Regression (Log-Loss, No Regularization)"
        )


def train_and_evaluate_logistic(
    reg_type: str = "none",
    student_input: Optional[Dict[str, Any]] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Trains and evaluates logistic regression models, computes Odds Ratios,
    generates Confusion Matrix plot, and performs live inference.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_type="preprocessed")

    active_features = [f for f in SUPERVISED_CLASSIFICATION_FEATURES if f in df.columns]
    target_col = CLASSIFICATION_TARGET if CLASSIFICATION_TARGET in df.columns else "Severity"

    # Convert multiclass Severity to binary if needed
    if target_col == "Severity" and df["Severity"].nunique() > 2:
        y = (df["Severity"] >= 3).astype(int)
    else:
        y = df[target_col].astype(int)

    X = df[active_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, model_name = get_logistic_model(reg_type)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_classification_model(y_train, y_train_pred)
    test_metrics = evaluate_classification_model(y_test, y_test_pred, y_prob=y_test_prob)

    intercept = round(float(model.intercept_[0]), 4)
    coefficients = []
    eq_terms = []
    for f, coef in zip(active_features, model.coef_[0]):
        c_val = round(float(coef), 4)
        odds_ratio = round(float(np.exp(c_val)), 3)
        is_zero = abs(c_val) == 0.0
        coefficients.append({
            "feature": f,
            "weight": c_val,
            "odds_ratio": odds_ratio,
            "is_zero": is_zero
        })
        sign = "+" if c_val >= 0 else "-"
        eq_terms.append(f"{sign} {abs(c_val)}*({f})")

    equation_str = f"logit(P) = {intercept} " + " ".join(eq_terms)

    # Confusion Matrix Visualization
    cm_plot = f"log_cm_{reg_type}.png"
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
    plt.title(f"Confusion Matrix ({model_name})", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / cm_plot)
    plt.savefig(CHARTS_DIR / cm_plot)
    plt.close()

    # Live Predictor
    prediction = None
    if student_input:
        prediction = predict_logistic_severity(model, student_input, active_features)

    # Optional model saving
    if save_model:
        model_filename = f"logistic_{reg_type}.pkl" if reg_type != "none" else "logistic_reg.pkl"
        joblib.dump(model, MODELS_DIR / model_filename)

    return {
        "reg_type": reg_type,
        "model_name": model_name,
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
        "intercept": intercept,
        "coefficients": coefficients,
        "equation": equation_str,
        "plots": {
            "cm_plot": cm_plot
        },
        "prediction": prediction,
        "features": active_features
    }


def predict_logistic_severity(
    model,
    input_data: Dict[str, Any],
    features: List[str]
) -> Optional[Dict[str, Any]]:
    """Infers severity tier and probability for single-sample user input."""
    try:
        row_vals = [float(input_data.get(f, 0.0)) for f in features]
        input_df = pd.DataFrame([row_vals], columns=features)
        pred_class = int(model.predict(input_df)[0])
        prob = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)
        return {
            "class": "Severe Accident" if pred_class == 1 else "Minor / Moderate Accident",
            "is_severe": pred_class == 1,
            "probability": prob
        }
    except Exception:
        return None
