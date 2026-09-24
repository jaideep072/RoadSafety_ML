"""
IQR Outlier Detection and Capping module for RoadSafety_ML.
Computes quartile thresholds strictly on the training partition and clips extreme sensor anomalies.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


def iqr_outlier_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    factor: float = 1.5
) -> Dict[str, Any]:
    """
    Performs comprehensive IQR outlier diagnostics across numeric features.
    """
    column_outliers = {}
    outlier_masks = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()

        if len(valid) == 0:
            column_outliers[col] = 0
            outlier_masks.append(pd.Series(False, index=df.index))
            continue

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask = (series < lower) | (series > upper)
        column_outliers[col] = int(mask.sum())
        outlier_masks.append(mask.fillna(False))

    if outlier_masks:
        combined = np.column_stack([m.values for m in outlier_masks])
        n_rows_with_outlier = int(combined.any(axis=1).sum())
    else:
        n_rows_with_outlier = 0

    n_outliers = int(sum(column_outliers.values()))
    max_outliers = max(column_outliers.values()) if column_outliers else 0

    return {
        "column_outliers": column_outliers,
        "n_outliers": n_outliers,
        "n_rows_with_outlier": n_rows_with_outlier,
        "max_outliers": max_outliers,
        "n_numeric_columns": len(column_outliers),
    }


def detect_and_cap_iqr_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: List[str],
    factor: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """
    Calculates IQR bounds on train_df and caps outliers in both train_df and test_df.
    Returns capped copies, list of summary records, and the bounds dictionary.
    """
    train_capped = train_df.copy()
    test_capped = test_df.copy()
    outliers_list: List[Dict[str, Any]] = []
    bounds_dict: Dict[str, Tuple[float, float]] = {}

    for col in columns:
        if col not in train_df.columns:
            continue

        q1 = float(train_df[col].quantile(0.25))
        q3 = float(train_df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        bounds_dict[col] = (lower_bound, upper_bound)

        before_train = train_capped[col].copy()
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)
        capped_count = int(np.sum(before_train != train_capped[col]))

        if test_df is not None and col in test_df.columns:
            test_capped[col] = np.clip(test_capped[col], lower_bound, upper_bound)

        outliers_list.append({
            "column": col,
            "bounds": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
            "capped": capped_count
        })

    return train_capped, test_capped, outliers_list, bounds_dict
