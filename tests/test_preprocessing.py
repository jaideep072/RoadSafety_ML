"""
Unit tests for preprocessing modules: missing value handling, imputation benchmarks,
IQR outlier filtering, categorical encoding, and feature scaling.
"""

import pytest
import pandas as pd
import numpy as np

from app.preprocessing.missing import analyze_missing_values
from app.preprocessing.imputation import compare_imputation_strategies, apply_median_imputation
from app.preprocessing.outliers import detect_and_cap_iqr_outliers, iqr_outlier_analysis
from app.preprocessing.encoding import (
    apply_one_hot_encoding,
    apply_ordinal_encoding,
    apply_target_encoding
)
from app.preprocessing.scaling import apply_standard_scaling, apply_minmax_scaling


@pytest.fixture
def sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "Temperature(F)": [65.0, 70.0, np.nan, 150.0, -40.0, 72.0, 68.0, 80.0, 55.0, 60.0],
        "Visibility(mi)": [10.0, 5.0, 2.0, 10.0, np.nan, 8.0, 9.0, 10.0, 7.0, 6.0],
        "Sunrise_Sunset": ["Day", "Night", "Day", "Day", "Night", "Day", "Night", "Day", "Night", "Day"],
        "Wind_Direction": ["N", "S", "E", "W", "N", "S", "E", "W", "N", "S"],
        "Severity": [2, 2, 3, 4, 1, 2, 3, 2, 2, 3]
    })
    return df


def test_analyze_missing_values(sample_data):
    res = analyze_missing_values(sample_data, ["Temperature(F)", "Visibility(mi)"])
    assert res["total_rows"] == 10
    assert "missing_cols" in res
    assert "rowdrop" in res
    assert "coldrop" in res
    assert "mean_imputation" in res
    assert "median_imputation" in res


def test_apply_median_imputation(sample_data):
    train_df = sample_data.iloc[:8].copy()
    test_df = sample_data.iloc[8:].copy()

    tr_clean, te_clean, imputer = apply_median_imputation(
        train_df, test_df, ["Temperature(F)", "Visibility(mi)"]
    )
    assert tr_clean["Temperature(F)"].isnull().sum() == 0
    assert te_clean["Visibility(mi)"].isnull().sum() == 0


def test_detect_and_cap_iqr_outliers(sample_data):
    train_df = sample_data.iloc[:8].copy()
    test_df = sample_data.iloc[8:].copy()

    tr_clean, te_clean, outliers_list, bounds = detect_and_cap_iqr_outliers(
        train_df, test_df, ["Temperature(F)"]
    )
    assert len(outliers_list) == 1
    assert "Temperature(F)" in bounds
    lower, upper = bounds["Temperature(F)"]
    assert tr_clean["Temperature(F)"].max() <= upper
    assert tr_clean["Temperature(F)"].min() >= lower


def test_apply_one_hot_encoding(sample_data):
    train_df = sample_data.iloc[:8].copy()
    test_df = sample_data.iloc[8:].copy()

    tr_ohe, te_ohe, ohe, features = apply_one_hot_encoding(
        train_df, test_df, ["Wind_Direction"]
    )
    assert len(tr_ohe) == 8
    assert len(te_ohe) == 2
    assert len(features) > 0


def test_apply_standard_and_minmax_scaling(sample_data):
    clean_df = sample_data.dropna().copy()
    train_df = clean_df.iloc[:5].copy()
    test_df = clean_df.iloc[5:].copy()

    # Standard Scaling
    tr_std, te_std, std_scaler = apply_standard_scaling(train_df, test_df, ["Temperature(F)"])
    assert abs(tr_std["Temperature(F)"].mean()) < 1e-5

    # MinMax Scaling
    tr_mm, te_mm, mm_scaler = apply_minmax_scaling(train_df, test_df, ["Temperature(F)"])
    assert tr_mm["Temperature(F)"].min() >= 0.0
    assert tr_mm["Temperature(F)"].max() <= 1.0
