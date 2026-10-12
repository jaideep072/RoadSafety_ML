"""
Unit tests for data loading, summary generation, and data validation routines.
"""

import pytest
import pandas as pd
import numpy as np

from app.data.loader import load_dataset, get_data_summary, generate_synthetic_dataset
from app.data.validators import (
    validate_dataset_schema,
    validate_clustering_data,
    validate_geographic_coordinates,
    DataValidationError
)


def test_load_dataset():
    df = load_dataset(dataset_type="original", nrows=500)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 500
    assert "Severity" in df.columns


def test_get_data_summary():
    summary = get_data_summary(dataset_type="original", nrows=200)
    assert "n_rows" in summary
    assert "n_cols" in summary
    assert "columns" in summary
    assert "missing_counts" in summary
    assert "preview" in summary
    assert summary["n_rows"] == 200


def test_validate_dataset_schema_success():
    df = pd.DataFrame({
        "Temperature(F)": [65.0, 72.0, 58.0],
        "Severity": [1, 2, 3]
    })
    is_valid, warnings = validate_dataset_schema(
        df,
        required_columns=["Temperature(F)", "Severity"],
        numeric_columns=["Temperature(F)"]
    )
    assert is_valid is True
    assert len(warnings) == 0


def test_validate_dataset_schema_missing_column():
    df = pd.DataFrame({"Temperature(F)": [65.0, 72.0]})
    with pytest.raises(DataValidationError):
        validate_dataset_schema(df, required_columns=["Missing_Column"])


def test_validate_clustering_data():
    df = pd.DataFrame({
        "Lat": [35.2, 36.1, np.nan, 34.8, 35.5],
        "Lng": [-80.8, -80.2, -81.0, -79.9, -80.5],
        "Temp": [70.0, 65.0, 60.0, 75.0, 80.0]
    })
    clean = validate_clustering_data(df, ["Lat", "Lng", "Temp"], min_rows=3)
    assert len(clean) == 4
    assert clean.isnull().sum().sum() == 0


def test_validate_geographic_coordinates():
    df = pd.DataFrame({
        "Start_Lat": [35.2271, 999.0, 40.7128, -50.0],
        "Start_Lng": [-80.8431, -74.0060, -74.0060, 200.0]
    })
    clean_geo = validate_geographic_coordinates(df, lat_col="Start_Lat", lng_col="Start_Lng")
    assert len(clean_geo) == 2
