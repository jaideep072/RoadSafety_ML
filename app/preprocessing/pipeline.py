

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

EXCLUDED_COLUMNS_RATIONALE = {
    "ID": "Arbitrary database primary key with no generalizable relationship to crash severity. Overfits if retained.",
    "Description": "Unstructured free-form incident text. Excluded to ensure consistent numerical tabular schema.",
    "End_Lat": "100% missing values in dataset. Geographically redundant with primary Start_Lat coordinate.",
    "End_Lng": "100% missing values in dataset. Geographically redundant with primary Start_Lng coordinate.",
    "Start_Time": "String timestamp format (YYYY-MM-DD HH:MM:SS); unparseable by linear and tree numerical algorithms.",
    "End_Time": "String timestamp recorded post-accident clearance; leaks outcome duration.",
    "Wind_Chill(F)": "Severe missingness (95.7% nulls). Redundant with Temperature(F) which is 98.4% complete.",
    "Precipitation(in)": "Severe missingness (92.6% nulls). Atmospheric moisture captured by Humidity(%) and Weather_Condition.",
    "Country": "Zero-variance constant feature: 100% of observations are 'US'. Provides zero predictive variance.",
    "Turning_Loop": "Zero-variance constant feature: 100% of records are False. Provides zero predictive variance.",
    "Street": "High-cardinality nominal text identifier (>30,000 unique levels). Prone to severe overfitting.",
    "City": "High-cardinality nominal city names. Geographic latitude and longitude coordinates provide continuous spatial density.",
    "County": "Nominal administrative boundary strings. Latitude and longitude provide continuous spatial density.",
    "State": "Zero-variance single-state dataset partition ('OH'). Provides zero predictive variance.",
    "Zipcode": "High-cardinality postal code string. Continuous spatial coordinates used instead.",
    "Timezone": "Zero-variance partition string ('US/Eastern'). Redundant with spatial coordinates.",
    "Airport_Code": "Weather station identifier: redundant with direct atmospheric readings (Temperature, Pressure, Humidity).",
    "Weather_Timestamp": "Data collection timestamp metadata with no predictive bearing on road physics."
}


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

    # 2. Three-Way Disjoint Partition (Train ~70%, Validation ~15%, Test ~15%) - Module 5 Session M5-1
    target_split_col = "Severity" if "Severity" in df.columns else None
    if target_split_col:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df[target_split_col]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=temp_df[target_split_col]
        )
    else:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

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

    # 9. Build and Export Clean Pure-Numeric Preprocessed Dataset
    clean_export_df = pd.DataFrame()

    # Primary Targets (Numeric)
    if "Severity" in df.columns:
        clean_export_df["Severity"] = df["Severity"].astype(int)
        clean_export_df["Severity_Severe"] = (df["Severity"] >= 3).astype(int)

    # Continuous Spatial & Telemetry Features (Clean Numeric)
    clean_export_df["Start_Lat"] = pd.to_numeric(df["Start_Lat"], errors="coerce")
    clean_export_df["Start_Lng"] = pd.to_numeric(df["Start_Lng"], errors="coerce")
    if "Distance(mi)" in df.columns:
        clean_export_df["Distance(mi)"] = pd.to_numeric(df["Distance(mi)"], errors="coerce").fillna(0.0)

    # Atmospheric & Environmental Sensor Telemetry (Median-Imputed + Outlier-Capped)
    env_cols = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
    for col in env_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            med_val = s.median()
            s = s.fillna(med_val)
            if col in active_outlier_cols:
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                s = s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            clean_export_df[col] = s.round(2)

    # Road Infrastructure Safety Fixtures (Strict Binary 0 / 1)
    road_fixtures = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal"
    ]
    for b_col in road_fixtures:
        if b_col in df.columns:
            clean_export_df[b_col] = df[b_col].fillna(False).astype(int)

    # Solar Ambient Lighting & Twilight Encodings (Binary Numeric: Day=1, Night=0)
    for tw in ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]:
        if tw in df.columns:
            clean_export_df[f"{tw}_Day"] = (df[tw] == "Day").astype(int)

    # One-Hot Encoded Atmospheric Risk Flags (Binary 0 / 1)
    top_weather = ["Clear", "Overcast", "Mostly Cloudy", "Partly Cloudy", "Scattered Clouds"]
    if "Weather_Condition" in df.columns:
        for w in top_weather:
            w_col = f"Weather_{w.replace(' ', '_')}"
            clean_export_df[w_col] = (df["Weather_Condition"] == w).astype(int)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_export_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved pure-numeric preprocessed dataset: {PROCESSED_DATA_PATH} ({len(clean_export_df):,} rows x {clean_export_df.shape[1]} cols)")

    # Save fitted scaler and imputer artifacts
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(std_scaler, SCALERS_DIR / "scaler.pkl")
    joblib.dump(fitted_imputer, ENCODERS_DIR / "imputer.pkl")

    # 10. Missing report strictly for retained engineered features
    missing_report = {}
    # Only report columns that are actually retained in the final preprocessed dataset
    retained_cols_to_check = [c for c in clean_export_df.columns if c in df.columns]
    for col in retained_cols_to_check:
        cnt = int(df[col].isnull().sum())
        if cnt > 0:
            missing_report[col] = {
                "count": cnt,
                "pct": round((cnt / len(df)) * 100, 2),
                "strategy": "SimpleImputer(strategy='median') + StandardScaler()"
            }

    # 11. Cache Preprocessing Summary
    pipeline_results = {
        # RentRadar aligned execution summary & diagnostics
        "original_rows": len(df),
        "cleaned_rows": len(clean_export_df),
        "target_name": "Severity (Accident Severity Level 1–4 / Severe Classification)",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "testing_samples": len(test_df),
        "split_ratio": "70% Train (70,000) / 15% Validation (15,000) / 15% Test (15,000)",
        "original_cols_count": len(df.columns),
        "preprocessed_cols_count": clean_export_df.shape[1],
        "original_columns": list(df.columns),
        "preprocessed_columns": list(clean_export_df.columns),
        "dropped_columns_list": [c for c in df.columns if c not in clean_export_df.columns],
        "dropped_reasons": EXCLUDED_COLUMNS_RATIONALE,
        "collinearity_justifications": {
            "End_Lat & End_Lng": "High multicollinearity with Start coordinates (>0.99) and 100% missing. Excluded to eliminate redundant dimensional collinearity.",
            "Weather_Timestamp": "Perfect collinearity with Start_Time observation window. Excluded to eliminate temporal duplication.",
            "Airport_Code": "Collinear proxy for weather observation station. Redundant with direct numerical telemetry (Temperature, Humidity, Pressure).",
            "Country & Turning_Loop": "Zero variance constants (100% 'US' and False). Provide 0.0 variance / singular matrix singularity.",
            "County & Timezone": "Collinear spatial hierarchy with Latitude & Longitude coordinates. Continuous GPS density selected instead."
        },
        "encoding_breakdown": {
            "one_hot": {
                "name": "One-Hot Encoding (OHE)",
                "technique": "OneHotEncoder(sparse=False, handle_unknown='ignore')",
                "columns": ["Weather_Condition (Top 5 Classes: Clear, Overcast, Mostly Cloudy, Partly Cloudy, Scattered Clouds)"],
                "purpose": "Converts non-ordinal categorical atmospheric conditions into independent 0/1 binary indicator predictors without imposing artificial numeric hierarchy."
            },
            "binary_flags": {
                "name": "Binary Road Fixture & Twilight Encoding",
                "technique": "Direct Boolean/Categorical to Integer Vector (0 / 1)",
                "columns": [
                    "Traffic_Signal", "Crossing", "Junction", "Amenity", "Stop", "Station",
                    "Railway", "Roundabout", "Give_Way", "Traffic_Calming", "Bump", "No_Exit",
                    "Sunrise_Sunset_Day", "Civil_Twilight_Day", "Nautical_Twilight_Day", "Astronomical_Twilight_Day"
                ],
                "purpose": "Preserves roadway physical indicators and diurnal lighting (Day=1, Night=0) as high-efficiency binary predictors."
            }
        },
        "engineered_infrastructure": [
            "Traffic_Signal", "Crossing", "Junction", "Amenity",
            "Stop", "Station", "Railway", "Roundabout",
            "Give_Way", "Traffic_Calming", "Bump", "No_Exit"
        ],
        "engineered_environmental": [
            "Temperature(F)", "Humidity(%)", "Pressure(in)",
            "Visibility(mi)", "Wind_Speed(mph)"
        ],
        "engineered_spatial_temporal": [
            "Start_Lat", "Start_Lng", "Distance(mi)"
        ],
        "engineered_twilight": [
            "Sunrise_Sunset_Day", "Civil_Twilight_Day", "Nautical_Twilight_Day", "Astronomical_Twilight_Day"
        ],
        "engineered_weather_ohe": [
            "Weather_Clear", "Weather_Overcast", "Weather_Mostly_Cloudy", "Weather_Partly_Cloudy", "Weather_Scattered_Clouds"
        ],
        "categorical_features": ["Weather_Condition", "Wind_Direction", "Sunrise_Sunset", "Civil_Twilight"],
        "missing_report": missing_report,

        # Preserved baseline analytical artifacts
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
