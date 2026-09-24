"""
Model Evaluation and Metrics package for RoadSafety_ML.
Computes rigorous, unfabricated performance metrics for classification, regression, and clustering.
"""

from app.evaluation.classification import evaluate_classification_model, ClassificationMetrics
from app.evaluation.regression import evaluate_regression_model, RegressionMetrics
from app.evaluation.clustering import evaluate_clustering_model, ClusteringMetrics

__all__ = [
    "evaluate_classification_model",
    "ClassificationMetrics",
    "evaluate_regression_model",
    "RegressionMetrics",
    "evaluate_clustering_model",
    "ClusteringMetrics"
]
