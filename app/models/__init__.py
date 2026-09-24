"""
Supervised Learning Models package for RoadSafety_ML.
Includes Linear Regression (OLS/Ridge/Lasso/ElasticNet), Logistic Regression,
Decision Tree, and Advanced Gradient Ensemble Classifiers & Regressors.
"""

from app.models.linear_regression import train_and_evaluate_linear, predict_linear_severity
from app.models.logistic_regression import train_and_evaluate_logistic, predict_logistic_severity
from app.models.decision_tree import train_and_evaluate_tree, predict_tree_severity
from app.models.ensembles import ALGO_CONFIG, train_ensemble_model

__all__ = [
    "train_and_evaluate_linear",
    "predict_linear_severity",
    "train_and_evaluate_logistic",
    "predict_logistic_severity",
    "train_and_evaluate_tree",
    "predict_tree_severity",
    "ALGO_CONFIG",
    "train_ensemble_model"
]
