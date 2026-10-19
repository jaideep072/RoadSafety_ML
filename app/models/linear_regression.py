"""
Linear Regression models module for RoadSafety_ML.
Implements Ordinary Least Squares (OLS), Ridge (L2 Penalty), Lasso (L1 Penalty),
and ElasticNet (Convex Combination) for continuous accident delay/distance prediction.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import joblib

from app.config import (
    CHARTS_DIR,
    STATIC_DIR,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    SUPERVISED_REGRESSION_FEATURES,
    REGRESSION_TARGET
)
from app.data.loader import load_dataset
from app.evaluation.regression import evaluate_regression_model


def get_linear_model(reg_type: str = "none"):
    """Returns initialized scikit-learn regressor based on regularization type."""
    reg_type = reg_type.lower()
    if reg_type == "ridge":
        return Ridge(alpha=1.0, random_state=42), "Ridge Regression (L2 Penalty)"
    elif reg_type == "lasso":
        return Lasso(alpha=0.01, random_state=42), "Lasso Regression (L1 Penalty)"
    elif reg_type == "elasticnet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), "ElasticNet Regression (L1 + L2 Penalty)"
    else:
        return LinearRegression(), "Standard Linear Regression (Ordinary Least Squares)"


def train_and_evaluate_linear(
    reg_type: str = "none",
    student_input: Optional[Dict[str, Any]] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Trains and evaluates linear regression models, generates residual diagnostics plots,
    and handles single-instance inference predictions.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_type="preprocessed")

    active_features = [f for f in SUPERVISED_REGRESSION_FEATURES if f in df.columns]
    target_col = REGRESSION_TARGET if REGRESSION_TARGET in df.columns else df.select_dtypes(include=[np.number]).columns[-1]

    X = df[active_features].copy()
    y = df[target_col].copy()

    # 3-Way Disjoint Partition (Train ~70%, Validation ~15%, Test ~15%) - Module 5 Session M5-1
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    model, model_name = get_linear_model(reg_type)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_metrics = evaluate_regression_model(y_train, y_train_pred, n_features=len(active_features))
    val_metrics = evaluate_regression_model(y_val, y_val_pred, n_features=len(active_features))
    test_metrics = evaluate_regression_model(y_test, y_test_pred, n_features=len(active_features))

    intercept = round(float(model.intercept_), 4)
    coefficients = []
    eq_terms = []
    for f, coef in zip(active_features, model.coef_):
        c_val = round(float(coef), 4)
        is_zero = abs(c_val) == 0.0
        coefficients.append({
            "feature": f,
            "weight": c_val,
            "is_zero": is_zero
        })
        sign = "+" if c_val >= 0 else "-"
        eq_terms.append(f"{sign} {abs(c_val)}*({f})")

    equation_str = f"{target_col} = {intercept} " + " ".join(eq_terms)

    # Visualization: Actual vs Predicted
    plot_actual_vs_pred = f"lr_actual_pred_{reg_type}.png"
    sample_size = min(500, len(y_test))
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)

    plt.figure(figsize=(6, 4.5), dpi=120)
    plt.scatter(y_test.iloc[sample_idx], y_test_pred[sample_idx], alpha=0.4, color="#1e3a8a", edgecolors="none")
    min_val = min(y_test.iloc[sample_idx].min(), y_test_pred[sample_idx].min())
    max_val = max(y_test.iloc[sample_idx].max(), y_test_pred[sample_idx].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.5)
    plt.xlabel(f"Actual {target_col}", fontsize=10)
    plt.ylabel(f"Predicted {target_col}", fontsize=10)
    plt.title(f"Actual vs Predicted ({model_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(STATIC_DIR / plot_actual_vs_pred)
    plt.savefig(CHARTS_DIR / plot_actual_vs_pred)
    plt.close()

    # Visualization: Residuals Distribution
    plot_residuals = f"lr_residuals_{reg_type}.png"
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6, 4.5), dpi=120)
    plt.hist(residuals, bins=40, color="#0284c7", edgecolor="#0369a1", alpha=0.7)
    plt.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual Error (Actual - Predicted)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title(f"Residual Error Distribution ({model_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(STATIC_DIR / plot_residuals)
    plt.savefig(CHARTS_DIR / plot_residuals)
    plt.close()

    # 3-Model Regularization Benchmark Comparison for Road Safety
    comparison_models = [
        {"key": "none", "name": "Ordinary Least Squares (OLS)", "regularization": "None (Unconstrained)"},
        {"key": "ridge", "name": "Ridge Regression", "regularization": "L2 Penalty (alpha=1.0)"},
        {"key": "lasso", "name": "Lasso Regression", "regularization": "L1 Penalty (alpha=0.01)"}
    ]
    benchmark_list = []
    for cm in comparison_models:
        mod, _ = get_linear_model(cm["key"])
        mod.fit(X_train, y_train)
        tr_p = mod.predict(X_train)
        te_p = mod.predict(X_test)
        tr_ev = evaluate_regression_model(y_train, tr_p, len(active_features))
        te_ev = evaluate_regression_model(y_test, te_p, len(active_features))
        zero_cnt = int(sum(1 for c in mod.coef_ if abs(c) < 1e-4))
        benchmark_list.append({
            "key": cm["key"],
            "name": cm["name"],
            "regularization": cm["regularization"],
            "train_r2": round(float(tr_ev.r2), 2),
            "test_r2": round(float(te_ev.r2), 2),
            "mae": round(float(te_ev.mae), 4),
            "rmse": round(float(te_ev.rmse), 4),
            "zero_features": zero_cnt,
            "overfitting_gap": round(float(tr_ev.r2 - te_ev.r2), 2)
        })

    # Single-Instance Live Prediction
    prediction = None
    if student_input:
        prediction = predict_linear_severity(model, student_input, active_features)

    # Optional model saving
    if save_model:
        model_filename = f"linear_{reg_type}.pkl" if reg_type != "none" else "linear_reg.pkl"
        joblib.dump(model, MODELS_DIR / model_filename)

    return {
        "reg_type": reg_type,
        "model_name": model_name,
        "total_train": len(y_train),
        "total_val": len(y_val),
        "total_test": len(y_test),
        "split_description": "70% Train / 15% Validation / 15% Test Split (Module 5)",
        "metrics": {
            "test_r2": test_metrics.r2,
            "val_r2": val_metrics.r2,
            "train_r2": train_metrics.r2,
            "adj_r2": test_metrics.adj_r2,
            "rmse": test_metrics.rmse,
            "val_rmse": val_metrics.rmse,
            "mae": test_metrics.mae,
            "mse": test_metrics.mse
        },
        "benchmark": benchmark_list,
        "lasso_zero_features": next((b["zero_features"] for b in benchmark_list if b["key"] == "lasso"), 0),
        "intercept": intercept,
        "coefficients": coefficients,
        "equation": equation_str,
        "plots": {
            "actual_vs_pred": plot_actual_vs_pred,
            "residuals": plot_residuals
        },
        "prediction": prediction,
        "features": active_features
    }


def predict_linear_severity(
    model,
    input_data: Dict[str, Any],
    features: List[str]
) -> Optional[float]:
    """Infers predicted continuous target value for a user input sample."""
    try:
        row_vals = [float(input_data.get(f, 0.0)) for f in features]
        input_df = pd.DataFrame([row_vals], columns=features)
        pred_val = model.predict(input_df)[0]
        return round(max(0.0, float(pred_val)), 3)
    except Exception:
        return None
