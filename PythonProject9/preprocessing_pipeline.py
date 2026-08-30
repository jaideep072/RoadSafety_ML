import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "US_Accidents_March23.csv")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "preprocessing_cache.json")

def _missing_value_analysis(df, numeric_cols):
    """
    Returns a dict with per-column missing stats and
    results of four handling strategies:
      1. row-wise deletion
      2. column-wise deletion
      3. mean imputation
      4. median imputation
    """
    total_rows = len(df)

    # Per-column missing info
    col_info = {}
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        col_info[col] = {
            "missing": n_miss,
            "pct":     round(n_miss / total_rows * 100, 4),
            "dtype":   str(df[col].dtype),
        }

    missing_cols = {c: v for c, v in col_info.items() if v["missing"] > 0}
    total_missing_cells = sum(v["missing"] for v in col_info.values())
    rows_with_any_missing = int(df.isnull().any(axis=1).sum())

    # ---- 1. Row-wise deletion ----
    df_rowdrop = df.dropna()
    rows_dropped  = total_rows - len(df_rowdrop)
    rowdrop_pct   = round(rows_dropped / total_rows * 100, 4)
    rowdrop_applicable = rows_dropped < total_rows * 0.05  # < 5% loss

    rowdrop_result = {
        "applicable":     rowdrop_applicable,
        "rows_before":    total_rows,
        "rows_after":     int(len(df_rowdrop)),
        "rows_dropped":   rows_dropped,
        "pct_dropped":    rowdrop_pct,
        "reason": (
            f"Only {rows_dropped} row(s) ({rowdrop_pct}%) contain missing values. "
            "Dropping them causes negligible data loss — row-wise deletion is safe."
            if rowdrop_applicable else
            f"{rows_dropped} rows ({rowdrop_pct}%) would be lost. "
            "This is too high a data loss for row-wise deletion to be appropriate."
        ),
        "preview": df_rowdrop.head(3).fillna("").astype(str).to_dict(orient="records"),
    }

    # ---- 2. Column-wise deletion ----
    THRESHOLD = 30.0
    cols_to_drop = [
        c for c, v in col_info.items()
        if v["missing"] > 0 and v["pct"] > THRESHOLD
    ]
    cols_not_dropped = [
        c for c, v in col_info.items()
        if v["missing"] > 0 and v["pct"] <= THRESHOLD
    ]
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
                "reason": (
                    f"Only {v['pct']:.2f}% missing ({v['missing']} value(s)). "
                    f"Below {THRESHOLD}% threshold — column retained, use imputation instead."
                )
            }

    coldrop_result = {
        "applicable":      coldrop_applicable,
        "threshold_pct":   THRESHOLD,
        "cols_dropped":    cols_to_drop,
        "cols_retained":   cols_not_dropped,
        "per_col_reasons": per_col_reasons,
        "reason": (
            f"Columns with >{THRESHOLD}% missing data: {cols_to_drop}. These were dropped." 
            if coldrop_applicable else
            f"No column exceeds the {THRESHOLD}% missing threshold. "
            "Column-wise deletion is NOT applied — all columns are retained. "
            "Use row-wise deletion or imputation to handle the small number of missing values."
        ),
    }

    # ---- 3. Mean imputation (numeric columns only) ----
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
                "column":    col,
                "n_filled":  n,
                "fill_value": mean_val,
                "applicable": True,
                "reason":    f"Numeric column — {n} missing value(s) replaced with mean ({mean_val})."
            })

    # Non-numeric cols with missing values
    for col, v in col_info.items():
        if v["missing"] > 0 and col not in numeric_cols:
            mean_results.append({
                "column":    col,
                "n_filled":  v["missing"],
                "fill_value": None,
                "applicable": False,
                "reason":    (
                    f"Non-numeric column (dtype: {v['dtype']}) — mean imputation is not applicable. "
                    "Use mode imputation or a placeholder string instead."
                )
            })

    mean_imputation = {
        "results": mean_results,
        "preview": df_mean[numeric_cols].head(5).round(4).to_dict(orient="records")
                   if numeric_cols else [],
    }

    # ---- 4. Median imputation (numeric columns only) ----
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
                "column":    col,
                "n_filled":  n,
                "fill_value": median_val,
                "applicable": True,
                "reason":    f"Numeric column — {n} missing value(s) replaced with median ({median_val})."
            })

    for col, v in col_info.items():
        if v["missing"] > 0 and col not in numeric_cols:
            median_results.append({
                "column":    col,
                "n_filled":  v["missing"],
                "fill_value": None,
                "applicable": False,
                "reason":    (
                    f"Non-numeric column (dtype: {v['dtype']}) — median imputation is not applicable. "
                    "Use mode imputation or a placeholder string instead."
                )
            })

    median_imputation = {
        "results": median_results,
        "preview": df_median[numeric_cols].head(5).round(4).to_dict(orient="records")
                   if numeric_cols else [],
    }

    return {
        "total_rows":          total_rows,
        "total_missing_cells": total_missing_cells,
        "rows_with_missing":   rows_with_any_missing,
        "col_info":            col_info,
        "missing_cols":        missing_cols,
        "rowdrop":             rowdrop_result,
        "coldrop":             coldrop_result,
        "mean_imputation":     mean_imputation,
        "median_imputation":   median_imputation,
    }


def _iqr_outlier_analysis(df, numeric_cols):
    """
    IQR outlier detection across all numeric columns.
    Returns per-column counts, total sum, and unique rows with any outlier.
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
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

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


def run_preprocessing_pipeline(force_run=False):
    # Check if cache exists to return instantly
    if not force_run and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Failed to read cache, calculating fresh:", e)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH, nrows=100000)
    
    # 2. Split train and test (80/20, stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Severity"]
    )
    
    missing_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)", "Humidity(%)"]
    
    # ==========================================
    # 1. missing.py Calculations
    # ==========================================
    missing_pct = train_df.isna().mean() * 100
    missing_count = train_df.isna().sum()
    has_missing = missing_count[missing_count > 0].sort_values(ascending=False)
    
    missing_before_dict = {}
    for col in has_missing.index:
        pct = missing_pct[col]
        count = int(missing_count[col])
        missing_before_dict[col] = f"{pct:.3f}% ({count} rows)"
    
    missing_by_sunset = train_df.groupby("Sunrise_Sunset")["Temperature(F)"].apply(
        lambda x: x.isna().mean() * 100
    ).to_dict()

    # Detailed missing value analysis from provided helper
    missing_analysis_res = _missing_value_analysis(df, missing_cols)
    
    # For speed, perform imputation comparison on a subset of 10,000 rows
    comp_sample = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    X_tr = comp_sample[missing_cols]
    y_tr = comp_sample["Severity"]
    
    test_comp_sample = test_df.sample(n=min(3000, len(test_df)), random_state=42)
    X_te = test_comp_sample[missing_cols]
    y_te = test_comp_sample["Severity"]
    
    # Strategy 1: Listwise Deletion
    train_complete = pd.concat([X_tr, y_tr], axis=1).dropna()
    X_tr_d = train_complete[missing_cols]
    y_tr_d = train_complete["Severity"]
    
    test_complete = pd.concat([X_te, y_te], axis=1).dropna()
    X_te_d = test_complete[missing_cols]
    y_te_d = test_complete["Severity"]
    
    model_d = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=150))
    ])
    model_d.fit(X_tr_d, y_tr_d)
    acc_d = model_d.score(X_te_d, y_te_d) * 100
    
    listwise_stats = {
        "before_rows": len(X_tr),
        "after_rows": len(X_tr_d),
        "discarded_pct": f"{(1 - len(X_tr_d)/len(X_tr))*100:.1f}%",
        "accuracy": f"{acc_d:.2f}%"
    }
    
    # Strategy 2: Median Imputation
    median_imputer = SimpleImputer(strategy="median")
    X_tr_med = pd.DataFrame(median_imputer.fit_transform(X_tr), columns=missing_cols)
    X_te_med = pd.DataFrame(median_imputer.transform(X_te), columns=missing_cols)
    
    model_med = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=150))
    ])
    model_med.fit(X_tr_med, y_tr)
    acc_med = model_med.score(X_te_med, y_te) * 100
    
    # Strategy 3: KNN Imputation (k=5)
    knn_imputer = KNNImputer(n_neighbors=5)
    X_tr_knn = pd.DataFrame(knn_imputer.fit_transform(X_tr), columns=missing_cols)
    X_te_knn = pd.DataFrame(knn_imputer.transform(X_te), columns=missing_cols)
    
    model_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=150))
    ])
    model_knn.fit(X_tr_knn, y_tr)
    acc_knn = model_knn.score(X_te_knn, y_te) * 100
    
    # Strategy 4: Median + Missing Indicator
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), missing_cols)
    ], remainder="passthrough")
    
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=150))
    ])
    pipe.fit(X_tr, y_tr)
    acc_ind = pipe.score(X_te, y_te) * 100
    
    imputation_comparison = [
        {"strategy": "Listwise Deletion", "rows": f"{len(X_tr_d):,}", "accuracy": f"{acc_d:.2f}%"},
        {"strategy": "Median Imputation", "rows": f"{len(X_tr_med):,}", "accuracy": f"{acc_med:.2f}%"},
        {"strategy": "KNN Imputation (k=5)", "rows": f"{len(X_tr_knn):,}", "accuracy": f"{acc_knn:.2f}%"},
        {"strategy": "Median + Missing Indicator", "rows": f"{len(X_tr):,}", "accuracy": f"{acc_ind:.2f}%"}
    ]

    # Pre-impute
    full_imputer = SimpleImputer(strategy="median")
    train_df[missing_cols] = full_imputer.fit_transform(train_df[missing_cols])
    test_df[missing_cols] = full_imputer.transform(test_df[missing_cols])
    
    # ==========================================
    # 3. outlier_Fix.py Calculations
    # ==========================================
    outlier_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"]
    outliers_list = []
    
    for col in outlier_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_cap = train_df[col].copy()
        train_df[col] = np.clip(train_df[col], lower_bound, upper_bound)
        capped_count = int(np.sum(before_cap != train_df[col]))
        
        outliers_list.append({
            "column": col,
            "bounds": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
            "capped": capped_count
        })

    # Detailed IQR outlier detection
    iqr_results = _iqr_outlier_analysis(train_complete, outlier_cols)

    # ==========================================
    # 4. one_hot_encoding.py & ordinal_encoding.py
    # ==========================================
    nominal_cols = ["Wind_Direction"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[nominal_cols].fillna("Missing"))
    ohe_features = list(ohe.get_feature_names_out(nominal_cols))

    # Actual Ordinal Encoding implementation
    ordinal_cols = ["Sunrise_Sunset", "Civil_Twilight"]
    for col in ordinal_cols:
        # Fill missing values
        mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "Night"
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)

    twilight_order = ["Night", "Day"]
    ord_enc = OrdinalEncoder(categories=[twilight_order, twilight_order], handle_unknown="use_encoded_value", unknown_value=-1)
    train_ord = ord_enc.fit_transform(train_df[ordinal_cols])
    test_ord = ord_enc.transform(test_df[ordinal_cols])
    
    train_ord_df = pd.DataFrame(train_ord, columns=ordinal_cols, index=train_df.index).round(4)
    test_ord_df = pd.DataFrame(test_ord, columns=ordinal_cols, index=test_df.index).round(4)

    # ==========================================
    # Target Encoding
    # ==========================================
    TARGET_COLS = ["Severity"]
    target_results = {}
    for target in TARGET_COLS:
        if target not in train_df.columns:
            continue
        target_results[target] = {}
        train_target = pd.to_numeric(train_df[target], errors="coerce")
        global_mean = float(train_target.mean())

        for col in nominal_cols:
            col_train = train_df[col].fillna("Missing").astype(str)
            col_test = test_df[col].fillna("Missing").astype(str)
            
            target_map = (
                train_df
                .assign(_target=train_target)
                .assign(_col=col_train)
                .groupby("_col")["_target"]
                .mean()
            )

            train_encoded = col_train.map(target_map).fillna(global_mean)
            test_encoded = col_test.map(target_map).fillna(global_mean)

            train_target_df = pd.DataFrame({f"{col}_target_{target}": train_encoded})
            test_target_df = pd.DataFrame({f"{col}_target_{target}": test_encoded})

            target_results[target][col] = {
                "mapping": target_map.round(4).to_dict(),
                "train_preview": train_target_df.head(5).round(4).to_dict(orient="records"),
                "test_preview": test_target_df.head(5).round(4).to_dict(orient="records")
            }

    # ==========================================
    # Embedding-Based Encoding
    # ==========================================
    embedding_results = {}
    for col in nominal_cols:
        col_train = train_df[col].fillna("Missing").astype(str)
        col_test = test_df[col].fillna("Missing").astype(str)
        
        categories = col_train.unique().tolist()
        category_to_id = {category: index for index, category in enumerate(categories)}

        train_ids = col_train.map(category_to_id).fillna(-1).astype(int)
        test_ids = col_test.map(category_to_id).fillna(-1).astype(int)

        embedding_results[col] = {
            "category_to_id": category_to_id,
            "embedding_input_dimension": len(category_to_id),
            "embedding_output_dimension": 3,
            "train_preview": pd.DataFrame({f"{col}_embedding_id": train_ids}).head(5).to_dict(orient="records"),
            "test_preview": pd.DataFrame({f"{col}_embedding_id": test_ids}).head(5).to_dict(orient="records")
        }

    # ==========================================
    # 5. Mini_Max_Scaling.py (Normalization)
    # ==========================================
    scale_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"]
    before_scaling = train_df[scale_cols].head(5).round(2).to_dict("records")
    
    min_max_scaler = MinMaxScaler()
    train_minmax = train_df.copy()
    train_minmax[scale_cols] = min_max_scaler.fit_transform(train_df[scale_cols])
    after_minmax = train_minmax[scale_cols].head(5).round(4).to_dict("records")
    
    # ==========================================
    # 6. Standard_Scalar.py (Standardization)
    # ==========================================
    std_scaler = StandardScaler()
    train_std = train_df.copy()
    train_std[scale_cols] = std_scaler.fit_transform(train_df[scale_cols])
    after_std = train_std[scale_cols].head(5).round(4).to_dict("records")

    # Save to cache
    pipeline_results = {
        "missing": {
            "per_column": missing_before_dict,
            "grouped": {str(k): f"{v:.2f}%" for k, v in missing_by_sunset.items()}
        },
        "missing_analysis": missing_analysis_res,
        "listwise": listwise_stats,
        "imputation": imputation_comparison,
        "outliers": outliers_list,
        "iqr_summary": iqr_results,
        "encoding": {
            "nominal_count": len(nominal_cols),
            "ohe_count": len(ohe_features),
            "ohe_features": ohe_features[:8]  # sample of 8 features
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
    
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(pipeline_results, f, indent=4)
        print("Preprocessing cache generated successfully!")
    except Exception as e:
        print("Failed to save cache:", e)
        
    return pipeline_results

if __name__ == "__main__":
    run_preprocessing_pipeline(force_run=True)
