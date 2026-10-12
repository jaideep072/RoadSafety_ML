"""
Imputation strategies and comparative benchmark studies for RoadSafety_ML.
Implements Listwise Deletion, Mean, Median, KNN, and Missing Indicator transformations
while strictly preventing data leakage across train/test splits.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def compare_imputation_strategies(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str = "Severity",
    sample_size: int = 10000
) -> List[Dict[str, Any]]:
    """
    Empirically benchmarks four missing data strategies on a controlled split:
    1. Listwise Deletion (row drop)
    2. Median Imputation
    3. KNN Imputation (k=5)
    4. Median + Missing Indicator
    """
    clean_df = df[numeric_cols + [target_col]].dropna(subset=[target_col]).copy()
    if len(clean_df) > sample_size:
        clean_df = clean_df.sample(n=sample_size, random_state=42)

    X = clean_df[numeric_cols]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # 1. Listwise Deletion
    train_complete = pd.concat([X_train, y_train], axis=1).dropna()
    X_tr_d = train_complete[numeric_cols]
    y_tr_d = train_complete[target_col]

    test_complete = pd.concat([X_test, y_test], axis=1).dropna()
    X_te_d = test_complete[numeric_cols]
    y_te_d = test_complete[target_col]

    if len(X_tr_d) > 0 and len(X_te_d) > 0:
        pipe_d = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=42))
        ])
        pipe_d.fit(X_tr_d, y_tr_d)
        acc_d = pipe_d.score(X_te_d, y_te_d) * 100.0
    else:
        acc_d = 0.0

    results.append({
        "strategy": "Listwise Deletion",
        "rows": f"{len(X_tr_d):,}",
        "accuracy": f"{acc_d:.2f}%"
    })

    # 2. Median Imputation
    med_imputer = SimpleImputer(strategy="median")
    X_tr_med = pd.DataFrame(med_imputer.fit_transform(X_train), columns=numeric_cols)
    X_te_med = pd.DataFrame(med_imputer.transform(X_test), columns=numeric_cols)

    pipe_med = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe_med.fit(X_tr_med, y_train)
    acc_med = pipe_med.score(X_te_med, y_test) * 100.0
    results.append({
        "strategy": "Median Imputation",
        "rows": f"{len(X_tr_med):,}",
        "accuracy": f"{acc_med:.2f}%"
    })

    # 3. KNN Imputation (k=5)
    knn_imputer = KNNImputer(n_neighbors=5)
    X_tr_knn = pd.DataFrame(knn_imputer.fit_transform(X_train), columns=numeric_cols)
    X_te_knn = pd.DataFrame(knn_imputer.transform(X_test), columns=numeric_cols)

    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe_knn.fit(X_tr_knn, y_train)
    acc_knn = pipe_knn.score(X_te_knn, y_test) * 100.0
    results.append({
        "strategy": "KNN Imputation (k=5)",
        "rows": f"{len(X_tr_knn):,}",
        "accuracy": f"{acc_knn:.2f}%"
    })

    # 4. Median + Missing Indicator
    ct = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric_cols)
    ], remainder="passthrough")

    pipe_ind = Pipeline([
        ("preprocess", ct),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe_ind.fit(X_train, y_train)
    acc_ind = pipe_ind.score(X_test, y_test) * 100.0
    results.append({
        "strategy": "Median + Missing Indicator",
        "rows": f"{len(X_train):,}",
        "accuracy": f"{acc_ind:.2f}%"
    })

    return results


def apply_median_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Fits SimpleImputer(strategy='median') strictly on train_df,
    then transforms both train_df and test_df to prevent data leakage.
    """
    imputer = SimpleImputer(strategy="median")
    
    train_df_clean = train_df.copy()
    test_df_clean = test_df.copy()

    cols_to_impute = [c for c in numeric_cols if c in train_df.columns]
    
    train_df_clean[cols_to_impute] = imputer.fit_transform(train_df[cols_to_impute])
    test_df_clean[cols_to_impute] = imputer.transform(test_df[cols_to_impute])

    return train_df_clean, test_df_clean, imputer
