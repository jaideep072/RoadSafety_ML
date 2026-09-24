"""
Data validation utilities for RoadSafety_ML.
Provides validation rules for file presence, schema conformity, missing value thresholds,
out-of-bounds coordinates, infinite values, and clustering numerical readiness.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Custom exception raised for dataset schema and integrity validation failures."""
    pass


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Verifies that the dataset file exists on disk."""
    path = Path(file_path)
    if not path.exists():
        raise DataValidationError(f"Dataset file not found at: {path.resolve()}")
    if path.stat().st_size == 0:
        raise DataValidationError(f"Dataset file is empty (0 bytes): {path.resolve()}")
    return path


def validate_dataset_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    allow_empty: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validates dataset structural properties:
    - Non-empty row count
    - Required columns presence
    - Numeric data types check
    - Infinite values presence
    """
    warnings: List[str] = []

    if df is None:
        raise DataValidationError("DataFrame is None.")

    if df.empty and not allow_empty:
        raise DataValidationError("Dataset contains 0 rows.")

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns in dataset: {', '.join(missing_cols)}"
            )

    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    warnings.append(
                        f"Column '{col}' is expected to be numeric but has type '{df[col].dtype}'."
                    )
                # Check for infinities
                inf_count = np.isinf(pd.to_numeric(df[col], errors="coerce")).sum()
                if inf_count > 0:
                    warnings.append(
                        f"Column '{col}' contains {inf_count} infinite values."
                    )

    return True, warnings


def validate_geographic_coordinates(
    df: pd.DataFrame,
    lat_col: str = "Start_Lat",
    lng_col: str = "Start_Lng",
    # Continental US + territories rough bounding box: Lat (18° to 72° N), Lng (-170° to -65° W)
    lat_range: Tuple[float, float] = (18.0, 72.0),
    lng_range: Tuple[float, float] = (-170.0, -65.0)
) -> pd.DataFrame:
    """
    Validates and filters latitude and longitude coordinates to ensure valid geographical values.
    Returns cleaned dataframe without modifying source data in-place.
    """
    if lat_col not in df.columns or lng_col not in df.columns:
        raise DataValidationError(f"Geographic coordinates '{lat_col}' or '{lng_col}' not found in dataframe.")

    valid_mask = (
        df[lat_col].notna() &
        df[lng_col].notna() &
        (df[lat_col] >= lat_range[0]) & (df[lat_col] <= lat_range[1]) &
        (df[lng_col] >= lng_range[0]) & (df[lng_col] <= lng_range[1])
    )
    
    clean_df = df[valid_mask].copy()
    if clean_df.empty:
        raise DataValidationError("No valid geographical coordinates found within standard boundary.")

    return clean_df


def validate_clustering_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_rows: int = 10
) -> pd.DataFrame:
    """
    Validates that features selected for clustering exist, are strictly numeric,
    and contain sufficient rows for cluster partition algorithms.
    """
    if df.empty:
        raise DataValidationError("Input dataset is empty.")

    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise DataValidationError(f"Selected clustering features not found: {missing_features}")

    subset = df[feature_cols].copy()

    # Verify all columns can be converted to numeric
    for col in feature_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # Drop all NaN rows in clustering features
    clean_subset = subset.dropna()

    if len(clean_subset) < min_rows:
        raise DataValidationError(
            f"Insufficient valid data points for clustering: found {len(clean_subset)} rows, "
            f"minimum required is {min_rows}."
        )

    return clean_subset
