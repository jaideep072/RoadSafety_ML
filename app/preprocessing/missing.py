"""
Missing value diagnostics and evaluation for RoadSafety_ML.
Computes column-wise and condition-grouped missing data metrics.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_missing_values(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Analyzes missing data patterns and compares:
    1. Row-wise (listwise) deletion impacts
    2. Column-wise deletion thresholds (>30%)
    3. Mean imputation feasibility
    4. Median imputation feasibility
    """
    total_rows = len(df)
    col_info = {}

    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        col_info[col] = {
            "missing": n_miss,
            "pct": round(n_miss / total_rows * 100, 4) if total_rows > 0 else 0.0,
            "dtype": str(df[col].dtype),
        }

    missing_cols = {c: v for c, v in col_info.items() if v["missing"] > 0}
    total_missing_cells = sum(v["missing"] for v in col_info.values())
    rows_with_any_missing = int(df.isnull().any(axis=1).sum())

    # 1. Row-wise (Listwise) Deletion
    df_rowdrop = df.dropna()
    rows_dropped = total_rows - len(df_rowdrop)
    rowdrop_pct = round(rows_dropped / total_rows * 100, 4) if total_rows > 0 else 0.0
    rowdrop_applicable = rows_dropped < total_rows * 0.05  # Safe if < 5% loss

    rowdrop_result = {
        "applicable": rowdrop_applicable,
        "rows_before": total_rows,
        "rows_after": int(len(df_rowdrop)),
        "rows_dropped": rows_dropped,
        "pct_dropped": rowdrop_pct,
        "reason": (
            f"Only {rows_dropped} row(s) ({rowdrop_pct}%) contain missing values. "
            "Dropping them causes minimal data loss."
            if rowdrop_applicable else
            f"{rows_dropped} rows ({rowdrop_pct}%) would be lost. "
            "Listwise deletion discards too much sample variance."
        ),
        "preview": df_rowdrop.head(3).fillna("").astype(str).to_dict(orient="records"),
    }

    # 2. Column-wise Deletion (30% threshold)
    THRESHOLD = 30.0
    cols_to_drop = [c for c, v in col_info.items() if v["missing"] > 0 and v["pct"] > THRESHOLD]
    cols_not_dropped = [c for c, v in col_info.items() if v["missing"] > 0 and v["pct"] <= THRESHOLD]
    coldrop_applicable = len(cols_to_drop) > 0

    per_col_reasons = {}
    for c, v in col_info.items():
        if v["missing"] == 0:
            continue
        if v["pct"] > THRESHOLD:
            per_col_reasons[c] = {
                "drop": True,
                "reason": f"{v['pct']:.2f}% missing — exceeds {THRESHOLD}% threshold. Column dropped."
            }
        else:
            per_col_reasons[c] = {
                "drop": False,
                "reason": f"Only {v['pct']:.2f}% missing ({v['missing']} cells). Retained for imputation."
            }

    coldrop_result = {
        "applicable": coldrop_applicable,
        "threshold_pct": THRESHOLD,
        "cols_dropped": cols_to_drop,
        "cols_retained": cols_not_dropped,
        "per_col_reasons": per_col_reasons,
        "reason": (
            f"Columns exceeding >{THRESHOLD}% threshold: {cols_to_drop}."
            if coldrop_applicable else
            f"No column exceeds the {THRESHOLD}% missing threshold — all features retained."
        )
    }

    # 3. Mean Imputation
    mean_results = []
    df_mean = df.copy()
    for col in numeric_cols:
        if col not in df_mean.columns:
            continue
        n = int(df_mean[col].isnull().sum())
        if n > 0:
            mean_val = round(float(df_mean[col].mean()), 4)
            df_mean[col] = df_mean[col].fillna(mean_val)
            mean_results.append({
                "column": col,
                "n_filled": n,
                "fill_value": mean_val,
                "applicable": True,
                "reason": f"Numeric column — {n} missing value(s) replaced with mean ({mean_val})."
            })

    # 4. Median Imputation
    median_results = []
    df_median = df.copy()
    for col in numeric_cols:
        if col not in df_median.columns:
            continue
        n = int(df_median[col].isnull().sum())
        if n > 0:
            median_val = round(float(df_median[col].median()), 4)
            df_median[col] = df_median[col].fillna(median_val)
            median_results.append({
                "column": col,
                "n_filled": n,
                "fill_value": median_val,
                "applicable": True,
                "reason": f"Numeric column — {n} missing value(s) replaced with median ({median_val})."
            })

    return {
        "total_rows": total_rows,
        "total_missing_cells": total_missing_cells,
        "rows_with_missing": rows_with_any_missing,
        "col_info": col_info,
        "missing_cols": missing_cols,
        "rowdrop": rowdrop_result,
        "coldrop": coldrop_result,
        "mean_imputation": {
            "results": mean_results,
            "preview": df_mean[numeric_cols].head(5).round(4).to_dict(orient="records") if numeric_cols else []
        },
        "median_imputation": {
            "results": median_results,
            "preview": df_median[numeric_cols].head(5).round(4).to_dict(orient="records") if numeric_cols else []
        }
    }
