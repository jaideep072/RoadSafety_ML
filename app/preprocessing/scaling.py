"""
Feature Scaling and Normalization module for RoadSafety_ML.
Implements StandardScaler (Z-Score) and MinMaxScaler ([0, 1] normalization)
without data leakage across training and testing partitions.
"""

from typing import List, Tuple, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def apply_standard_scaling(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Fits StandardScaler on train_df[columns] and transforms train_df and test_df.
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    cols = [c for c in columns if c in train_df.columns]

    train_scaled[cols] = scaler.fit_transform(train_df[cols])
    if test_scaled is not None:
        test_scaled[cols] = scaler.transform(test_df[cols])

    return train_scaled, test_scaled, scaler


def apply_minmax_scaling(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MinMaxScaler]:
    """
    Fits MinMaxScaler on train_df[columns] and transforms train_df and test_df.
    """
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    cols = [c for c in columns if c in train_df.columns]

    train_scaled[cols] = scaler.fit_transform(train_df[cols])
    if test_scaled is not None:
        test_scaled[cols] = scaler.transform(test_df[cols])

    return train_scaled, test_scaled, scaler
