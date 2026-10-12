"""
RoadSafety_ML Preprocessing Package.
Modular transformations for missing data diagnostics, imputation benchmarks,
IQR outlier filtering, categorical encoding, and feature scaling.
"""

from app.preprocessing.missing import analyze_missing_values
from app.preprocessing.imputation import compare_imputation_strategies, apply_median_imputation
from app.preprocessing.outliers import detect_and_cap_iqr_outliers, iqr_outlier_analysis
from app.preprocessing.encoding import (
    apply_one_hot_encoding,
    apply_ordinal_encoding,
    apply_target_encoding,
    apply_embedding_encoding
)
from app.preprocessing.scaling import apply_standard_scaling, apply_minmax_scaling
from app.preprocessing.pipeline import run_preprocessing_pipeline

__all__ = [
    "analyze_missing_values",
    "compare_imputation_strategies",
    "apply_median_imputation",
    "detect_and_cap_iqr_outliers",
    "iqr_outlier_analysis",
    "apply_one_hot_encoding",
    "apply_ordinal_encoding",
    "apply_target_encoding",
    "apply_embedding_encoding",
    "apply_standard_scaling",
    "apply_minmax_scaling",
    "run_preprocessing_pipeline"
]
