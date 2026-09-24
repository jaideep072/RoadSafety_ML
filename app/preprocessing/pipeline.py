"""
Unified Data Preprocessing Pipeline orchestrator for RoadSafety_ML.
Executes missing value analysis, imputation benchmarks, IQR capping,
categorical encodings, feature scaling, and stores preprocessed datasets and cache.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

from app.config import (
    PROCESSED_DATA_PATH,
    PREPROCESSING_CACHE_PATH,
    SCALERS_DIR,
    ENCODERS_DIR,
    DEFAULT_NUMERIC_FEATURES
)
from app.data.loader import load_dataset
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

logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    force_run: bool = False,
    nrows: int = 100000
) -> Dict[str, Any]:
    """
    Executes the end-to-end preprocessing workflow and outputs summary cache and preprocessed CSV.
    """
    if not force_run and PREPROCESSING_CACHE_PATH.exists() and PROCESSED_DATA_PATH.exists():
        try:
            with open(PREPROCESSING_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read preprocessing cache, regenerating: {e}")

    logger.info("Executing comprehensive Data Preprocessing Pipeline...")

    # 1. Load Raw Dataset
    df = load_dataset(dataset_type="original", nrows=nrows)

    # 2. Stratified Train/Test Split (80/20)
    target_split_col = "Severity" if "Severity" in df.columns else None
    if target_split_col:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[target_split_col]
        )
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    missing_cols = [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]

    # 3. Missing Value Analysis & Diagnostics
    missing_pct = train_df.isna().mean() * 100
    missing_count = train_df.isna().sum()
    has_missing = missing_count[missing_count > 0].sort_values(ascending=False)

    missing_before_dict = {}
    for col in has_missing.index:
        pct = missing_pct[col]
        cnt = int(missing_count[col])
        missing_before_dict[col] = f"{pct:.3f}% ({cnt} rows)"

    missing_by_sunset = {}
    if "Sunrise_Sunset" in train_df.columns and "Temperature(F)" in train_df.columns:
        missing_by_sunset = train_df.groupby("Sunrise_Sunset")["Temperature(F)"].apply(
            lambda x: x.isna().mean() * 100
        ).to_dict()

    missing_analysis_res = analyze_missing_values(df, missing_cols)

    # 4. Imputation Benchmarking
    imputation_comparison = compare_imputation_strategies(
        df=df,
        numeric_cols=missing_cols,
        target_col="Severity" if "Severity" in df.columns else missing_cols[0],
        sample_size=10000
    )

    # 5. Fit Median Imputation (Leakage-free)
    train_df, test_df, fitted_imputer = apply_median_imputation(train_df, test_df, missing_cols)

    # 6. IQR Outlier Capping
    outlier_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"]
    active_outlier_cols = [c for c in outlier_cols if c in train_df.columns]
    train_df, test_df, outliers_list, _ = detect_and_cap_iqr_outliers(
        train_df=train_df,
        test_df=test_df,
        columns=active_outlier_cols,
        factor=1.5
    )
    iqr_summary = iqr_outlier_analysis(train_df, active_outlier_cols)

    # 7. Categorical Encodings
    nominal_cols = [c for c in ["Wind_Direction"] if c in train_df.columns]
    ohe_features = []
    if nominal_cols:
        train_ohe, test_ohe, fitted_ohe, ohe_features = apply_one_hot_encoding(
            train_df, test_df, nominal_cols
        )
    else:
        train_ohe = pd.DataFrame()

    ordinal_cols = [c for c in ["Sunrise_Sunset", "Civil_Twilight"] if c in train_df.columns]
    twilight_order = ["Night", "Day"]
    train_ord_df = pd.DataFrame()
    test_ord_df = pd.DataFrame()
    if ordinal_cols:
        categories = [twilight_order for _ in ordinal_cols]
        train_ord_df, test_ord_df, fitted_ord = apply_ordinal_encoding(
            train_df, test_df, ordinal_cols, categories
        )

    target_results = apply_target_encoding(train_df, test_df, nominal_cols, target_col="Severity")
    embedding_results = apply_embedding_encoding(train_df, test_df, nominal_cols)

    # 8. Feature Scaling Previews
    scale_cols = [c for c in ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"] if c in train_df.columns]
    before_scaling = train_df[scale_cols].head(5).round(2).to_dict("records") if scale_cols else []

    train_minmax, _, minmax_scaler = apply_minmax_scaling(train_df, None, scale_cols)
    after_minmax = train_minmax[scale_cols].head(5).round(4).to_dict("records") if scale_cols else []

    train_std, _, std_scaler = apply_standard_scaling(train_df, None, scale_cols)
    after_std = train_std[scale_cols].head(5).round(4).to_dict("records") if scale_cols else []

    # 9. Build and Export Clean Preprocessed Dataset
    clean_export_df = df.copy()
    # Impute numeric features
    for col in missing_cols:
        if col in clean_export_df.columns:
            clean_export_df[col] = clean_export_df[col].fillna(clean_export_df[col].median())
    
    # Cap outliers
    for col in active_outlier_cols:
        q1 = clean_export_df[col].quantile(0.25)
        q3 = clean_export_df[col].quantile(0.75)
        iqr = q3 - q1
        clean_export_df[col] = clean_export_df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # Create binary Severity_Severe target (Severity >= 3)
    if "Severity" in clean_export_df.columns:
        clean_export_df["Severity_Severe"] = (clean_export_df["Severity"] >= 3).astype(int)

    # Encode Sunrise_Sunset as binary numeric (Day=1, Night=0)
    if "Sunrise_Sunset" in clean_export_df.columns:
        clean_export_df["Sunrise_Sunset_Day"] = (clean_export_df["Sunrise_Sunset"] == "Day").astype(int)

    # Convert boolean road features to int
    for b_col in ["Crossing", "Junction", "Traffic_Signal", "Amenity", "Bump", "Railway", "Station", "Stop"]:
        if b_col in clean_export_df.columns:
            clean_export_df[b_col] = clean_export_df[b_col].fillna(False).astype(int)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_export_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved preprocessed dataset: {PROCESSED_DATA_PATH} ({len(clean_export_df):,} rows)")

    # Save fitted scaler and imputer artifacts
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(std_scaler, SCALERS_DIR / "scaler.pkl")
    joblib.dump(fitted_imputer, ENCODERS_DIR / "imputer.pkl")

    # 10. Cache Preprocessing Summary
    pipeline_results = {
        "missing": {
            "per_column": missing_before_dict,
            "grouped": {str(k): f"{v:.2f}%" for k, v in missing_by_sunset.items()}
        },
        "missing_analysis": missing_analysis_res,
        "listwise": imputation_comparison[0] if imputation_comparison else {},
        "imputation": imputation_comparison,
        "outliers": outliers_list,
        "iqr_summary": iqr_summary,
        "encoding": {
            "nominal_count": len(nominal_cols),
            "ohe_count": len(ohe_features),
            "ohe_features": ohe_features[:8]
        },
        "ordinal": {
            "columns": ordinal_cols,
            "train_preview": train_ord_df.head(5).to_dict(orient="records"),
            "test_preview": test_ord_df.head(5).to_dict(orient="records"),
            "mappings": {
                col: {cat: float(i) for i, cat in enumerate(twilight_order)} for col in ordinal_cols
            }
        },
        "target_encoding": target_results,
        "embedding": embedding_results,
        "minmax": {
            "columns": scale_cols,
            "before": before_scaling,
            "after": after_minmax
        },
        "standard": {
            "columns": scale_cols,
            "before": before_scaling,
            "after": after_std
        }
    }

    PREPROCESSING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(pipeline_results, f, indent=4)
    logger.info(f"Saved preprocessing cache: {PREPROCESSING_CACHE_PATH}")

    return pipeline_results
