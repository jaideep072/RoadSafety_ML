"""
Unified Machine Learning Training and Benchmarking Pipeline for RoadSafety_ML.
Trains all 10+ supervised classification and regression models, evaluates metrics,
and serializes artifacts into the artifacts/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib

from app.config import (
    MODELS_DIR,
    SCALERS_DIR,
    MODELS_CACHE_PATH,
    SUPERVISED_REGRESSION_FEATURES,
    SUPERVISED_CLASSIFICATION_FEATURES,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET
)
from app.data.loader import load_dataset
from app.evaluation.regression import evaluate_regression_model
from app.evaluation.classification import evaluate_classification_model

logger = logging.getLogger(__name__)


def run_models_pipeline(force_run: bool = False) -> Dict[str, Any]:
    """
    Executes model training for the full 10+ algorithm suite and persists metrics cache.
    """
    if not force_run and MODELS_CACHE_PATH.exists():
        try:
            with open(MODELS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read models cache, regenerating: {e}")

    logger.info("Starting Full Machine Learning Model Training & Benchmarking Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_type="preprocessed")

    # 1. Setup Datasets
    reg_features = [f for f in SUPERVISED_REGRESSION_FEATURES if f in df.columns]
    clf_features = [f for f in SUPERVISED_CLASSIFICATION_FEATURES if f in df.columns]

    X_reg = df[reg_features].copy()
    y_reg = df[REGRESSION_TARGET].copy() if REGRESSION_TARGET in df.columns else df.select_dtypes(include=[np.number]).iloc[:, 0]

    X_clf = df[clf_features].copy()
    if CLASSIFICATION_TARGET in df.columns:
        y_clf = df[CLASSIFICATION_TARGET].astype(int)
    else:
        y_clf = (df["Severity"] >= 3).astype(int)

    # Train/Test Splits
    X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_clf_tr, X_clf_te, y_clf_tr, y_clf_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # 2. Train and Evaluate Regression Models
    regression_models = {
        "linear_ols": (LinearRegression(), "linear_reg.pkl", "Linear Regression (OLS)"),
        "linear_ridge": (Ridge(alpha=1.0, random_state=42), "linear_ridge.pkl", "Ridge Regression (L2)"),
        "linear_lasso": (Lasso(alpha=0.01, random_state=42), "linear_lasso.pkl", "Lasso Regression (L1)"),
        "linear_elasticnet": (ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), "linear_elasticnet.pkl", "ElasticNet Regression"),
        "dt_regressor": (DecisionTreeRegressor(max_depth=8, random_state=42), "dt_regressor.pkl", "Decision Tree Regressor"),
        "rf_regressor": (RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1), "rf_regressor.pkl", "Random Forest Regressor"),
        "gb_regressor": (GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42), "gb_regressor.pkl", "Gradient Boosting Regressor"),
        "xgb_regressor": (XGBRegressor(n_estimators=30, max_depth=4, random_state=42), "xgb_regressor.pkl", "XGBoost Regressor"),
        "lgbm_regressor": (LGBMRegressor(n_estimators=30, max_depth=4, random_state=42, verbose=-1), "lgbm_regressor.pkl", "LightGBM Regressor")
    }

    regression_results = []
    for key, (model, filename, name) in regression_models.items():
        logger.info(f"Training Regressor: {name}...")
        model.fit(X_reg_tr, y_reg_tr)
        y_pred = model.predict(X_reg_te)
        metrics = evaluate_regression_model(y_reg_te, y_pred, n_features=len(reg_features))
        joblib.dump(model, MODELS_DIR / filename)

        regression_results.append({
            "model_key": key,
            "name": name,
            "filename": filename,
            "r2": metrics.r2,
            "adj_r2": metrics.adj_r2,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "mse": metrics.mse
        })

    # 3. Train and Evaluate Classification Models
    classification_models = {
        "logistic_ols": (LogisticRegression(penalty=None, max_iter=1000, random_state=42), "logistic_reg.pkl", "Logistic Regression (OLS)"),
        "logistic_ridge": (LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42), "logistic_ridge.pkl", "Logistic Ridge (L2)"),
        "logistic_lasso": (LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42), "logistic_lasso.pkl", "Logistic Lasso (L1)"),
        "logistic_elasticnet": (LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=1000, random_state=42), "logistic_elasticnet.pkl", "Logistic ElasticNet"),
        "dt_classifier": (DecisionTreeClassifier(max_depth=6, random_state=42), "dt_classifier.pkl", "Decision Tree Classifier"),
        "rf_classifier": (RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1), "rf_classifier.pkl", "Random Forest Classifier"),
        "adaboost_classifier": (AdaBoostClassifier(n_estimators=30, random_state=42), "adaboost_classifier.pkl", "AdaBoost Classifier"),
        "gb_classifier": (GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42), "gb_classifier.pkl", "Gradient Boosting Classifier"),
        "xgb_classifier": (XGBClassifier(n_estimators=30, max_depth=4, random_state=42, eval_metric="logloss"), "xgb_classifier.pkl", "XGBoost Classifier"),
        "lgbm_classifier": (LGBMClassifier(n_estimators=30, max_depth=4, random_state=42, verbose=-1), "lgbm_classifier.pkl", "LightGBM Classifier")
    }

    classification_results = []
    for key, (model, filename, name) in classification_models.items():
        logger.info(f"Training Classifier: {name}...")
        model.fit(X_clf_tr, y_clf_tr)
        y_pred = model.predict(X_clf_te)
        
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_clf_te)[:, 1]
            except Exception:
                y_prob = None

        metrics = evaluate_classification_model(y_clf_te, y_pred, y_prob=y_prob)
        joblib.dump(model, MODELS_DIR / filename)

        classification_results.append({
            "model_key": key,
            "name": name,
            "filename": filename,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "tp": metrics.tp,
            "tn": metrics.tn,
            "fp": metrics.fp,
            "fn": metrics.fn
        })

    # Save Pipeline Results to Cache
    models_summary = {
        "regression_leaderboard": regression_results,
        "classification_leaderboard": classification_results,
        "n_models_trained": len(regression_results) + len(classification_results),
        "regression_features": reg_features,
        "classification_features": clf_features
    }

    MODELS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(models_summary, f, indent=4)
    logger.info(f"Models cache saved to: {MODELS_CACHE_PATH}")

    return models_summary
