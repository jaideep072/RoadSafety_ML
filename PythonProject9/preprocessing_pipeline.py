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

DATA_PATH = r"C:\Users\yeswanth sai\PycharmProjects\PythonProject9\data\US_Accidents_Sample_100k.csv"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "preprocessing_cache.json")

def run_preprocessing_pipeline(force_run=False):
    # Check if cache exists to return instantly
    if not force_run and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Failed to read cache, calculating fresh:", e)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
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
    
    # ==========================================
    # 2. listwisedeletion.py & imputation_comparison.py Calculations
    # ==========================================
    X_tr = train_df[missing_cols]
    y_tr = train_df["Severity"]
    X_te = test_df[missing_cols]
    y_te = test_df["Severity"]
    
    # Strategy 1: Listwise Deletion
    train_complete = pd.concat([X_tr, y_tr], axis=1).dropna()
    X_tr_d = train_complete[missing_cols]
    y_tr_d = train_complete["Severity"]
    
    test_complete = pd.concat([X_te, y_te], axis=1).dropna()
    X_te_d = test_complete[missing_cols]
    y_te_d = test_complete["Severity"]
    
    model_d = LogisticRegression(max_iter=500)
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
    
    model_med = LogisticRegression(max_iter=500)
    model_med.fit(X_tr_med, y_tr)
    acc_med = model_med.score(X_te_med, y_te) * 100
    
    # Strategy 3: KNN Imputation (k=5)
    knn_imputer = KNNImputer(n_neighbors=5)
    X_tr_knn = pd.DataFrame(knn_imputer.fit_transform(X_tr), columns=missing_cols)
    X_te_knn = pd.DataFrame(knn_imputer.transform(X_te), columns=missing_cols)
    
    model_knn = LogisticRegression(max_iter=500)
    model_knn.fit(X_tr_knn, y_tr)
    acc_knn = model_knn.score(X_te_knn, y_te) * 100
    
    # Strategy 4: Median + Missing Indicator
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), missing_cols)
    ], remainder="passthrough")
    
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=500))
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

    # ==========================================
    # 4. one_hot_encoding.py & ordinal_encoding.py
    # ==========================================
    nominal_cols = ["Wind_Direction"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[nominal_cols])
    ohe_features = list(ohe.get_feature_names_out(nominal_cols))
    
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
        "listwise": listwise_stats,
        "imputation": imputation_comparison,
        "outliers": outliers_list,
        "encoding": {
            "nominal_count": len(nominal_cols),
            "ohe_count": len(ohe_features),
            "ohe_features": ohe_features[:8]  # sample of 8 features
        },
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
