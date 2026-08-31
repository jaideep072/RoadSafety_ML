import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_PATH = os.path.join(BASE_DIR, "static", "models_cache.json")

# Model pickling paths
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")

# Subtypes of models
MODEL_PATHS = {
    # Regression Page (Standard OLS)
    "linear_ols": os.path.join(MODELS_DIR, "linear_reg.pkl"),
    "logistic_ols": os.path.join(MODELS_DIR, "logistic_reg.pkl"),
    # Regularization Page
    "linear_ridge": os.path.join(MODELS_DIR, "linear_ridge.pkl"),
    "linear_lasso": os.path.join(MODELS_DIR, "linear_lasso.pkl"),
    "linear_elasticnet": os.path.join(MODELS_DIR, "linear_elasticnet.pkl"),
    "logistic_ridge": os.path.join(MODELS_DIR, "logistic_ridge.pkl"),
    "logistic_lasso": os.path.join(MODELS_DIR, "logistic_lasso.pkl"),
    "logistic_elasticnet": os.path.join(MODELS_DIR, "logistic_elasticnet.pkl"),
    # Decision Tree Page
    "linear_dt": os.path.join(MODELS_DIR, "dt_regressor.pkl"),
    "logistic_dt": os.path.join(MODELS_DIR, "dt_classifier.pkl")
}

# Features and targets
FEATURE_COLS = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
LINEAR_TARGET = "Distance(mi)"
LOGISTIC_TARGET = "Severity"

def serialize_tree(estimator, feature_names):
    """
    Serializes a scikit-learn DecisionTree into a nested dictionary representation.
    """
    tree_ = estimator.tree_
    def recurse(node_id):
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        
        # Check if leaf node
        if left_child == right_child:
            val = tree_.value[node_id]
            if len(val.shape) == 3:  # Classifier
                class_counts = val[0][0].tolist()
                pred_class = int(np.argmax(class_counts))
                pred_val = pred_class
            else:
                class_counts = None
                pred_val = float(val[0][0])
            return {
                "type": "leaf",
                "id": int(node_id),
                "prediction": pred_val,
                "class_counts": class_counts,
                "impurity": round(float(tree_.impurity[node_id]), 4),
                "samples": int(tree_.n_node_samples[node_id])
            }
        else:
            feat_idx = tree_.feature[node_id]
            feat_name = feature_names[feat_idx]
            threshold = round(float(tree_.threshold[node_id]), 4)
            return {
                "type": "split",
                "id": int(node_id),
                "feature": feat_name,
                "threshold": threshold,
                "impurity": round(float(tree_.impurity[node_id]), 4),
                "samples": int(tree_.n_node_samples[node_id]),
                "left": recurse(left_child),
                "right": recurse(right_child)
            }
    return recurse(0)

def train_models(force_run=False):
    """
    Trains all 10 models (OLS, Ridge, Lasso, Elastic Net, Decision Trees).
    Saves validation metrics, feature importances, and coefficients to models_cache.json.
    Saves trained estimators (models, scaler, imputer) to models/ for inference.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    # Check if cache and model files exist to return early
    all_exist = os.path.exists(CACHE_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)
    if all_exist:
        for path in MODEL_PATHS.values():
            if not os.path.exists(path):
                all_exist = False
                break
    
    if not force_run and all_exist:
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if "decision_tree" in cache_data and "tree_structure" in cache_data["decision_tree"]["linear"]:
                    return cache_data
        except Exception as e:
            print("Failed to read models cache, retraining:", e)

    # 1. Load Data
    from load_data import load_data
    df = load_data(DATA_PATH)
    
    # Use 20,000 row sample to keep SAGA/elasticnet solver fast
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()

    # Subset required columns
    all_needed_cols = FEATURE_COLS + [LINEAR_TARGET, LOGISTIC_TARGET]
    df_sample = df_sample[all_needed_cols].copy()

    # Drop target null values
    df_sample = df_sample.dropna(subset=[LINEAR_TARGET, LOGISTIC_TARGET])

    # Convert targets to appropriate types
    df_sample[LINEAR_TARGET] = pd.to_numeric(df_sample[LINEAR_TARGET], errors="coerce")
    df_sample[LOGISTIC_TARGET] = pd.to_numeric(df_sample[LOGISTIC_TARGET], errors="coerce").astype(int)

    # Drop rows if targets are still null
    df_sample = df_sample.dropna(subset=[LINEAR_TARGET, LOGISTIC_TARGET])

    X = df_sample[FEATURE_COLS]
    y_linear = df_sample[LINEAR_TARGET]
    y_logistic = df_sample[LOGISTIC_TARGET]

    # Split train/test
    X_train, X_test, y_lin_train, y_lin_test, y_log_train, y_log_test = train_test_split(
        X, y_linear, y_logistic, test_size=0.2, random_state=42, stratify=y_logistic
    )

    # 2. Fit Imputer
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLS)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLS)

    # 3. Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=FEATURE_COLS)

    # ==========================================
    # A. REGRESSION PAGE MODELS (Standard OLS)
    # ==========================================
    # Linear Regression (OLS)
    lin_ols = LinearRegression()
    lin_ols.fit(X_train_scaled, y_lin_train)
    y_lin_pred_train = lin_ols.predict(X_train_scaled)
    y_lin_pred_test = lin_ols.predict(X_test_scaled)
    
    lin_ols_coefs = {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_ols.coef_)}
    lin_ols_intercept = round(float(lin_ols.intercept_), 4)

    equation_parts = []
    for col, coef in lin_ols_coefs.items():
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f"{sign} {abs(coef)}*[{col}_scaled]")
    equation_str = f"Distance = {lin_ols_intercept} " + " ".join(equation_parts)

    lin_predictions_preview = []
    preview_indices = np.random.choice(len(y_lin_test), min(10, len(y_lin_test)), replace=False)
    for idx in preview_indices:
        lin_predictions_preview.append({
            "actual": round(float(y_lin_test.iloc[idx]), 3),
            "predicted": round(float(y_lin_pred_test[idx]), 3)
        })

    # Logistic Regression (OLS / No penalty)
    log_ols = LogisticRegression(max_iter=1000, penalty=None)
    log_ols.fit(X_train_scaled, y_log_train)
    y_log_pred_train = log_ols.predict(X_train_scaled)
    y_log_pred_test = log_ols.predict(X_test_scaled)

    classes = sorted(list(log_ols.classes_))
    ols_precision, ols_recall, ols_f1, _ = precision_recall_fscore_support(y_log_test, y_log_pred_test, average="weighted")
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(y_log_test, y_log_pred_test, labels=classes)
    
    log_class_metrics = {}
    for i, c in enumerate(classes):
        log_class_metrics[str(c)] = {
            "precision": round(float(class_precision[i]), 4),
            "recall": round(float(class_recall[i]), 4),
            "f1": round(float(class_f1[i]), 4),
            "support": int(class_support[i])
        }

    log_ols_coefs = {}
    for class_idx, class_label in enumerate(classes):
        log_ols_coefs[str(class_label)] = {
            col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_ols.coef_[class_idx])
        }
    log_ols_intercepts = {str(c): round(float(inter), 4) for c, inter in zip(classes, log_ols.intercept_)}
    conf_mat_ols = confusion_matrix(y_log_test, y_log_pred_test, labels=classes)

    # ==========================================
    # B. REGULARIZATION PAGE MODELS (Ridge, Lasso, Elastic Net)
    # ==========================================
    # 1. Ridge Regression (L2)
    lin_ridge = Ridge(alpha=1.0)
    lin_ridge.fit(X_train_scaled, y_lin_train)
    y_ridge_pred = lin_ridge.predict(X_test_scaled)

    log_ridge = LogisticRegression(max_iter=1000, penalty='l2')
    log_ridge.fit(X_train_scaled, y_log_train)
    y_log_ridge_pred = log_ridge.predict(X_test_scaled)

    # 2. Lasso Regression (L1)
    lin_lasso = Lasso(alpha=0.01, max_iter=2000)
    lin_lasso.fit(X_train_scaled, y_lin_train)
    y_lasso_pred = lin_lasso.predict(X_test_scaled)

    log_lasso = LogisticRegression(max_iter=2000, penalty='l1', solver='saga')
    log_lasso.fit(X_train_scaled, y_log_train)
    y_log_lasso_pred = log_lasso.predict(X_test_scaled)

    # 3. Elastic Net (L1 + L2)
    lin_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    lin_en.fit(X_train_scaled, y_lin_train)
    y_en_pred = lin_en.predict(X_test_scaled)

    log_en = LogisticRegression(max_iter=2000, penalty='elasticnet', solver='saga', l1_ratio=0.5)
    log_en.fit(X_train_scaled, y_log_train)
    y_log_en_pred = log_en.predict(X_test_scaled)

    # Regularization Coefficients Shrinkage Comparisons
    # Linear coefficients comparison
    lin_coefs_compare = {
        "ols": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_ols.coef_)},
        "ridge": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_ridge.coef_)},
        "lasso": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_lasso.coef_)},
        "elasticnet": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_en.coef_)}
    }

    # Logistic coefficients comparison (for severity class 2 as a representative or overall average magnitude)
    # We can save class 2 coefficients or average magnitude of coefficients
    log_coefs_compare = {
        "ols": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_ols.coef_[idx])} for idx, c in enumerate(classes)},
        "ridge": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_ridge.coef_[idx])} for idx, c in enumerate(classes)},
        "lasso": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_lasso.coef_[idx])} for idx, c in enumerate(classes)},
        "elasticnet": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_en.coef_[idx])} for idx, c in enumerate(classes)}
    }

    # Regularization Metrics comparison
    linear_metrics_compare = {
        "ols": {
            "r2_test": round(r2_score(y_lin_test, y_lin_pred_test), 4),
            "mse": round(mean_squared_error(y_lin_test, y_lin_pred_test), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_lin_pred_test)), 4)
        },
        "ridge": {
            "r2_test": round(r2_score(y_lin_test, y_ridge_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_ridge_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_ridge_pred)), 4)
        },
        "lasso": {
            "r2_test": round(r2_score(y_lin_test, y_lasso_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_lasso_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_lasso_pred)), 4)
        },
        "elasticnet": {
            "r2_test": round(r2_score(y_lin_test, y_en_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_en_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_en_pred)), 4)
        }
    }

    def calc_log_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1_val, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        return {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1_val, 4)
        }

    logistic_metrics_compare = {
        "ols": calc_log_metrics(y_log_test, y_log_pred_test),
        "ridge": calc_log_metrics(y_log_test, y_log_ridge_pred),
        "lasso": calc_log_metrics(y_log_test, y_log_lasso_pred),
        "elasticnet": calc_log_metrics(y_log_test, y_log_en_pred)
    }

    # ==========================================
    # C. DECISION TREE PAGE MODELS
    # ==========================================
    # Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_reg.fit(X_train_imputed, y_lin_train) # Decision Trees don't require scaling!
    y_dt_pred_train = dt_reg.predict(X_train_imputed)
    y_dt_pred_test = dt_reg.predict(X_test_imputed)

    dt_reg_r2_train = r2_score(y_lin_train, y_dt_pred_train)
    dt_reg_r2_test = r2_score(y_lin_test, y_dt_pred_test)
    dt_reg_mse = mean_squared_error(y_lin_test, y_dt_pred_test)
    dt_reg_rmse = np.sqrt(dt_reg_mse)
    
    dt_reg_importances = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLS, dt_reg.feature_importances_)}

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train_imputed, y_log_train)
    y_dt_clf_pred_train = dt_clf.predict(X_train_imputed)
    y_dt_clf_pred_test = dt_clf.predict(X_test_imputed)

    dt_clf_acc_train = accuracy_score(y_log_train, y_dt_clf_pred_train)
    dt_clf_acc_test = accuracy_score(y_log_test, y_dt_clf_pred_test)
    dt_clf_precision, dt_clf_recall, dt_clf_f1, _ = precision_recall_fscore_support(y_log_test, y_dt_clf_pred_test, average="weighted")
    dt_clf_class_precision, dt_clf_class_recall, dt_clf_class_f1, dt_clf_class_support = precision_recall_fscore_support(y_log_test, y_dt_clf_pred_test, labels=classes)
    
    dt_clf_class_metrics = {}
    for i, c in enumerate(classes):
        dt_clf_class_metrics[str(c)] = {
            "precision": round(float(dt_clf_class_precision[i]), 4),
            "recall": round(float(dt_clf_class_recall[i]), 4),
            "f1": round(float(dt_clf_class_f1[i]), 4),
            "support": int(dt_clf_class_support[i])
        }

    dt_clf_importances = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLS, dt_clf.feature_importances_)}
    conf_mat_dt = confusion_matrix(y_log_test, y_dt_clf_pred_test, labels=classes)

    # ==========================================
    # SAVE ALL ESTIMATORS TO DISK
    # ==========================================
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(IMPUTER_PATH, "wb") as f:
        pickle.dump(imputer, f)

    estimators = {
        "linear_ols": lin_ols,
        "logistic_ols": log_ols,
        "linear_ridge": lin_ridge,
        "linear_lasso": lin_lasso,
        "linear_elasticnet": lin_en,
        "logistic_ridge": log_ridge,
        "logistic_lasso": log_lasso,
        "logistic_elasticnet": log_en,
        "linear_dt": dt_reg,
        "logistic_dt": dt_clf
    }

    for name, est in estimators.items():
        with open(MODEL_PATHS[name], "wb") as f:
            pickle.dump(est, f)

    # ==========================================
    # CACHE ALL METRICS TO JSON
    # ==========================================
    results = {
        "success": True,
        "sample_size": sample_size,
        "features": FEATURE_COLS,
        # Regression Page metrics
        "regression": {
            "linear": {
                "target": LINEAR_TARGET,
                "r2_train": round(r2_score(y_lin_train, y_lin_pred_train), 4),
                "r2_test": round(r2_score(y_lin_test, y_lin_pred_test), 4),
                "mse": round(mean_squared_error(y_lin_test, y_lin_pred_test), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_lin_pred_test)), 4),
                "coefficients": lin_ols_coefs,
                "intercept": lin_ols_intercept,
                "equation": equation_str,
                "preview": lin_predictions_preview
            },
            "logistic": {
                "target": LOGISTIC_TARGET,
                "classes": [int(c) for c in classes],
                "accuracy_train": round(accuracy_score(y_log_train, y_log_pred_train), 4),
                "accuracy_test": round(accuracy_score(y_log_test, y_log_pred_test), 4),
                "precision_weighted": round(ols_precision, 4),
                "recall_weighted": round(ols_recall, 4),
                "f1_weighted": round(ols_f1, 4),
                "class_metrics": log_class_metrics,
                "coefficients": log_ols_coefs,
                "intercepts": log_ols_intercepts,
                "confusion_matrix": conf_mat_ols.tolist()
            }
        },
        # Regularization Page comparisons
        "regularization": {
            "linear_metrics": linear_metrics_compare,
            "logistic_metrics": logistic_metrics_compare,
            "linear_coefs": lin_coefs_compare,
            "logistic_coefs": log_coefs_compare
        },
        # Decision Tree Page metrics
        "decision_tree": {
            "linear": {
                "target": LINEAR_TARGET,
                "r2_train": round(dt_reg_r2_train, 4),
                "r2_test": round(dt_reg_r2_test, 4),
                "mse": round(dt_reg_mse, 4),
                "rmse": round(dt_reg_rmse, 4),
                "feature_importances": dt_reg_importances,
                "max_depth": 5,
                "tree_structure": serialize_tree(dt_reg, FEATURE_COLS)
            },
            "logistic": {
                "target": LOGISTIC_TARGET,
                "classes": [int(c) for c in classes],
                "accuracy_train": round(dt_clf_acc_train, 4),
                "accuracy_test": round(dt_clf_acc_test, 4),
                "precision_weighted": round(dt_clf_precision, 4),
                "recall_weighted": round(dt_clf_recall, 4),
                "f1_weighted": round(dt_clf_f1, 4),
                "class_metrics": dt_clf_class_metrics,
                "feature_importances": dt_clf_importances,
                "confusion_matrix": conf_mat_dt.tolist(),
                "max_depth": 5,
                "tree_structure": serialize_tree(dt_clf, FEATURE_COLS)
            }
        }
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results

def predict_sample(features_dict, category="regression", model_type="none"):
    """
    Performs real-time predictions using the selected dashboard category and model subtype.
    category: 'regression', 'regularization', or 'decision_tree'
    model_type: 'none' (OLS), 'l2' (Ridge), 'l1' (Lasso), 'elasticnet'
    """
    # Formulate key
    if category == "decision_tree":
        linear_key = "linear_dt"
        logistic_key = "logistic_dt"
    else:
        # category is regression or regularization
        sub = "ols" if model_type == "none" else model_type
        linear_key = f"linear_{sub}"
        logistic_key = f"logistic_{sub}"

    # Verify model files exist
    needed_files = [SCALER_PATH, IMPUTER_PATH, MODEL_PATHS[linear_key], MODEL_PATHS[logistic_key]]
    if not all(os.path.exists(p) for p in needed_files):
        # Force re-training if any files are missing
        train_models(force_run=True)

    # Load models
    with open(MODEL_PATHS[linear_key], "rb") as f:
        lin_model = pickle.load(f)
    with open(MODEL_PATHS[logistic_key], "rb") as f:
        log_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)

    # Construct input DataFrame
    input_data = pd.DataFrame([features_dict], columns=FEATURE_COLS)
    
    # Handle Scaling (Decision Tree models don't use the scaled features!)
    if category == "decision_tree":
        input_processed = pd.DataFrame(imputer.transform(input_data), columns=FEATURE_COLS)
    else:
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=FEATURE_COLS)
        input_processed = pd.DataFrame(scaler.transform(input_imputed), columns=FEATURE_COLS)

    # Predict
    pred_distance = float(lin_model.predict(input_processed)[0])
    pred_severity = int(log_model.predict(input_processed)[0])
    
    # Class probabilities for logistic classifiers
    pred_probs = log_model.predict_proba(input_processed)[0]
    classes = [int(c) for c in log_model.classes_]
    probs_dict = {str(c): round(float(p), 4) for c, p in zip(classes, pred_probs)}

    return {
        "linear_prediction": {
            "target": LINEAR_TARGET,
            "predicted_value": max(0.0, round(pred_distance, 4))
        },
        "logistic_prediction": {
            "target": LOGISTIC_TARGET,
            "predicted_class": pred_severity,
            "probabilities": probs_dict
        }
    }

if __name__ == "__main__":
    print("Training models...")
    res = train_models(force_run=True)
    print("Training complete!")
    print("Linear R2 OLS:", res["regression"]["linear"]["r2_test"])
    print("Linear R2 Lasso:", res["regularization"]["linear_metrics"]["lasso"]["r2_test"])
    print("Decision Tree Accuracy:", res["decision_tree"]["logistic"]["accuracy_test"])
