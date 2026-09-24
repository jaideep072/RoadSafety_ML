"""
Categorical Feature Encoding module for RoadSafety_ML.
Implements One-Hot Encoding, Ordinal Encoding, Target Encoding, and Embedding Identifiers.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def apply_one_hot_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nominal_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Fits OneHotEncoder(drop='first', handle_unknown='ignore') on train_df and transforms both sets.
    """
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    
    train_subset = train_df[nominal_cols].fillna("Missing")
    test_subset = test_df[nominal_cols].fillna("Missing") if test_df is not None else None

    train_ohe = ohe.fit_transform(train_subset)
    feature_names = list(ohe.get_feature_names_out(nominal_cols))
    
    train_ohe_df = pd.DataFrame(train_ohe, columns=feature_names, index=train_df.index)

    if test_subset is not None:
        test_ohe = ohe.transform(test_subset)
        test_ohe_df = pd.DataFrame(test_ohe, columns=feature_names, index=test_df.index)
    else:
        test_ohe_df = pd.DataFrame(columns=feature_names)

    return train_ohe_df, test_ohe_df, ohe, feature_names


def apply_ordinal_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ordinal_cols: List[str],
    categories: List[List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """
    Fits OrdinalEncoder on specified ordered categorical dimensions.
    """
    ord_enc = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_subset = train_df[ordinal_cols].copy()
    test_subset = test_df[ordinal_cols].copy() if test_df is not None else None

    for col in ordinal_cols:
        mode_val = train_subset[col].mode()[0] if not train_subset[col].mode().empty else categories[0][0]
        train_subset[col] = train_subset[col].fillna(mode_val)
        if test_subset is not None:
            test_subset[col] = test_subset[col].fillna(mode_val)

    train_ord = ord_enc.fit_transform(train_subset)
    train_ord_df = pd.DataFrame(train_ord, columns=ordinal_cols, index=train_df.index).round(4)

    if test_subset is not None:
        test_ord = ord_enc.transform(test_subset)
        test_ord_df = pd.DataFrame(test_ord, columns=ordinal_cols, index=test_df.index).round(4)
    else:
        test_ord_df = pd.DataFrame(columns=ordinal_cols)

    return train_ord_df, test_ord_df, ord_enc


def apply_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nominal_cols: List[str],
    target_col: str = "Severity"
) -> Dict[str, Any]:
    """
    Computes smoothed target encoding based on training label statistics.
    """
    target_results: Dict[str, Any] = {}
    if target_col not in train_df.columns:
        return target_results

    train_target = pd.to_numeric(train_df[target_col], errors="coerce")
    global_mean = float(train_target.mean())

    for col in nominal_cols:
        if col not in train_df.columns:
            continue
        col_train = train_df[col].fillna("Missing").astype(str)
        col_test = test_df[col].fillna("Missing").astype(str) if test_df is not None else pd.Series([], dtype=str)

        target_map = (
            train_df
            .assign(_target=train_target)
            .assign(_col=col_train)
            .groupby("_col")["_target"]
            .mean()
        )

        train_encoded = col_train.map(target_map).fillna(global_mean)
        test_encoded = col_test.map(target_map).fillna(global_mean) if test_df is not None else pd.Series([])

        train_target_df = pd.DataFrame({f"{col}_target_{target_col}": train_encoded})
        test_target_df = pd.DataFrame({f"{col}_target_{target_col}": test_encoded})

        target_results[col] = {
            "mapping": target_map.round(4).to_dict(),
            "train_preview": train_target_df.head(5).round(4).to_dict(orient="records"),
            "test_preview": test_target_df.head(5).round(4).to_dict(orient="records")
        }

    return target_results


def apply_embedding_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nominal_cols: List[str]
) -> Dict[str, Any]:
    """
    Generates categorical index mappings for neural network embedding layer consumption.
    """
    embedding_results: Dict[str, Any] = {}

    for col in nominal_cols:
        if col not in train_df.columns:
            continue
        col_train = train_df[col].fillna("Missing").astype(str)
        col_test = test_df[col].fillna("Missing").astype(str) if test_df is not None else pd.Series([], dtype=str)

        categories = sorted(col_train.unique().tolist())
        category_to_id = {category: index for index, category in enumerate(categories)}

        train_ids = col_train.map(category_to_id).fillna(-1).astype(int)
        test_ids = col_test.map(category_to_id).fillna(-1).astype(int) if test_df is not None else pd.Series([], dtype=int)

        embedding_results[col] = {
            "category_to_id": category_to_id,
            "embedding_input_dimension": len(category_to_id),
            "embedding_output_dimension": min(50, (len(category_to_id) + 1) // 2),
            "train_preview": pd.DataFrame({f"{col}_embedding_id": train_ids}).head(5).to_dict(orient="records"),
            "test_preview": pd.DataFrame({f"{col}_embedding_id": test_ids}).head(5).to_dict(orient="records")
        }

    return embedding_results
