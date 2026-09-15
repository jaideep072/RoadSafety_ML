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
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "data", "roadsafety_preprocessed.csv")
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "US_Accidents_Sample_100k.csv")
if not os.path.exists(ORIGINAL_DATA_PATH):
    ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
DATA_PATH = PREPROCESSED_PATH if os.path.exists(PREPROCESSED_PATH) else ORIGINAL_DATA_PATH
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
    # 7 Tree Based Algorithms
    # 1. Decision Tree
    "linear_decision_tree": os.path.join(MODELS_DIR, "dt_regressor.pkl"),
    "logistic_decision_tree": os.path.join(MODELS_DIR, "dt_classifier.pkl"),
    # 2. Random Forest
    "linear_random_forest": os.path.join(MODELS_DIR, "rf_regressor.pkl"),
    "logistic_random_forest": os.path.join(MODELS_DIR, "rf_classifier.pkl"),
    # 3. AdaBoost
    "linear_adaboost": os.path.join(MODELS_DIR, "adaboost_regressor.pkl"),
    "logistic_adaboost": os.path.join(MODELS_DIR, "adaboost_classifier.pkl"),
    # 4. Gradient Boosting
    "linear_gradient_boosting": os.path.join(MODELS_DIR, "gb_regressor.pkl"),
    "logistic_gradient_boosting": os.path.join(MODELS_DIR, "gb_classifier.pkl"),
    # 5. XGBoost
    "linear_xgboost": os.path.join(MODELS_DIR, "xgb_regressor.pkl"),
    "logistic_xgboost": os.path.join(MODELS_DIR, "xgb_classifier.pkl"),
    # 6. LightGBM / Light Boost
    "linear_lightgbm": os.path.join(MODELS_DIR, "lgbm_regressor.pkl"),
    "logistic_lightgbm": os.path.join(MODELS_DIR, "lgbm_classifier.pkl"),
    # 7. CatBoost
    "linear_catboost": os.path.join(MODELS_DIR, "catboost_regressor.pkl"),
    "logistic_catboost": os.path.join(MODELS_DIR, "catboost_classifier.pkl")
}

# Features and targets
FEATURE_COLS = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
LINEAR_TARGET = "Distance(mi)"
LOGISTIC_TARGET = "Severity"

TREE_ALGO_NAMES = [
    ("decision_tree", "Decision Tree", "Single partitioning tree that recursively splits weather features."),
    ("random_forest", "Random Forest", "Bagging ensemble of diverse decision trees with feature sub-sampling."),
    ("adaboost", "AdaBoost", "Adaptive boosting focusing iteratively on previously misclassified accident instances."),
    ("gradient_boosting", "Gradient Boosting", "Sequentially adds decision trees optimizing pseudo-residuals."),
    ("xgboost", "XGBoost (Extreme Gradient Boosting)", "High-performance regularized gradient boosting with second-order gradients."),
    ("lightgbm", "Light Boost (LightGBM)", "Fast histogram-based gradient boosting with leaf-wise tree growth."),
    ("catboost", "CatBoost", "Symmetric oblivious decision trees with robust gradient estimation.")
]

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
                pred_val = pred_class + 1 # 1-indexed severity
            else:
                class_counts = None
                pred_val = float(val[0][0])
            return {
                "type": "leaf",
                "id": int(node_id),
                "prediction": round(pred_val, 4) if isinstance(pred_val, float) else pred_val,
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
    Trains all models:
    - Linear & Logistic (OLS, Ridge, Lasso, ElasticNet)
    - 7 Tree-Based Algorithms (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost)
    Saves validation metrics, feature importances, leaderboard benchmarks, and serialized models.
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
                if "tree_models" in cache_data and "algorithms" in cache_data["tree_models"]:
                    return cache_data
        except Exception as e:
            print("Failed to read models cache, retraining:", e)

    print("Training all machine learning models across pipeline...")

    # 1. Load Data
    from load_data import load_data
    df = load_data(DATA_PATH)
    
    # Use 20,000 row sample to keep SAGA/elasticnet & boosting models fast and responsive
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

    classes = sorted(list(y_log_train.unique()))
    # 0-indexed targets for XGBoost/LightGBM compatibility
    y_log_train_0 = y_log_train - 1
    y_log_test_0 = y_log_test - 1

    # ==========================================
    # A. LINEAR & LOGISTIC REGRESSION (OLS & REGULARIZATION)
    # ==========================================
    # 1. Linear OLS
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

    # 2. Logistic OLS (No penalty)
    log_ols = LogisticRegression(max_iter=1000, penalty=None)
    log_ols.fit(X_train_scaled, y_log_train)
    y_log_pred_train = log_ols.predict(X_train_scaled)
    y_log_pred_test = log_ols.predict(X_test_scaled)

    ols_precision, ols_recall, ols_f1, _ = precision_recall_fscore_support(y_log_test, y_log_pred_test, average="weighted", zero_division=0)
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(y_log_test, y_log_pred_test, labels=classes, zero_division=0)
    
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

    # 3. Ridge Regression (L2)
    lin_ridge = Ridge(alpha=1.0)
    lin_ridge.fit(X_train_scaled, y_lin_train)
    y_ridge_pred = lin_ridge.predict(X_test_scaled)

    log_ridge = LogisticRegression(max_iter=1000, penalty='l2')
    log_ridge.fit(X_train_scaled, y_log_train)
    y_log_ridge_pred = log_ridge.predict(X_test_scaled)

    # 4. Lasso Regression (L1)
    lin_lasso = Lasso(alpha=0.01, max_iter=2000)
    lin_lasso.fit(X_train_scaled, y_lin_train)
    y_lasso_pred = lin_lasso.predict(X_test_scaled)

    log_lasso = LogisticRegression(max_iter=2000, penalty='l1', solver='saga')
    log_lasso.fit(X_train_scaled, y_log_train)
    y_log_lasso_pred = log_lasso.predict(X_test_scaled)

    # 5. Elastic Net (L1 + L2)
    lin_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    lin_en.fit(X_train_scaled, y_lin_train)
    y_en_pred = lin_en.predict(X_test_scaled)

    log_en = LogisticRegression(max_iter=2000, penalty='elasticnet', solver='saga', l1_ratio=0.5)
    log_en.fit(X_train_scaled, y_log_train)
    y_log_en_pred = log_en.predict(X_test_scaled)

    # Regularization Comparisons
    lin_coefs_compare = {
        "ols": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_ols.coef_)},
        "ridge": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_ridge.coef_)},
        "lasso": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_lasso.coef_)},
        "elasticnet": {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, lin_en.coef_)}
    }

    log_coefs_compare = {
        "ols": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_ols.coef_[idx])} for idx, c in enumerate(classes)},
        "ridge": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_ridge.coef_[idx])} for idx, c in enumerate(classes)},
        "lasso": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_lasso.coef_[idx])} for idx, c in enumerate(classes)},
        "elasticnet": {str(c): {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLS, log_en.coef_[idx])} for idx, c in enumerate(classes)}
    }

    linear_metrics_compare = {
        "ols": {
            "r2_train": round(r2_score(y_lin_train, y_lin_pred_train), 4),
            "r2_test": round(r2_score(y_lin_test, y_lin_pred_test), 4),
            "mse": round(mean_squared_error(y_lin_test, y_lin_pred_test), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_lin_pred_test)), 4)
        },
        "ridge": {
            "r2_train": round(r2_score(y_lin_train, lin_ridge.predict(X_train_scaled)), 4),
            "r2_test": round(r2_score(y_lin_test, y_ridge_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_ridge_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_ridge_pred)), 4)
        },
        "lasso": {
            "r2_train": round(r2_score(y_lin_train, lin_lasso.predict(X_train_scaled)), 4),
            "r2_test": round(r2_score(y_lin_test, y_lasso_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_lasso_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_lasso_pred)), 4)
        },
        "elasticnet": {
            "r2_train": round(r2_score(y_lin_train, lin_en.predict(X_train_scaled)), 4),
            "r2_test": round(r2_score(y_lin_test, y_en_pred), 4),
            "mse": round(mean_squared_error(y_lin_test, y_en_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_lin_test, y_en_pred)), 4)
        }
    }

    def calc_log_metrics(y_true, y_pred, y_train_true=None, y_train_pred=None):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1_val, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        res = {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1_val, 4)
        }
        if y_train_true is not None and y_train_pred is not None:
            res["accuracy_train"] = round(accuracy_score(y_train_true, y_train_pred), 4)
        return res

    logistic_metrics_compare = {
        "ols": calc_log_metrics(y_log_test, y_log_pred_test, y_log_train, y_log_pred_train),
        "ridge": calc_log_metrics(y_log_test, y_log_ridge_pred, y_log_train, log_ridge.predict(X_train_scaled)),
        "lasso": calc_log_metrics(y_log_test, y_log_lasso_pred, y_log_train, log_lasso.predict(X_train_scaled)),
        "elasticnet": calc_log_metrics(y_log_test, y_log_en_pred, y_log_train, log_en.predict(X_train_scaled))
    }

    # ==========================================
    # B. 7 TREE-BASED ALGORITHMS
    # ==========================================
    # Instantiate models
    tree_regressors = {
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        "adaboost": AdaBoostRegressor(n_estimators=50, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
        "xgboost": XGBRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1, verbosity=0),
        "lightgbm": LGBMRegressor(n_estimators=50, max_depth=4, random_state=42, verbose=-1),
        "catboost": CatBoostRegressor(iterations=50, depth=4, random_seed=42, verbose=0)
    }

    tree_classifiers = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        "adaboost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
        "xgboost": XGBClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1, verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1),
        "catboost": CatBoostClassifier(iterations=50, depth=4, random_seed=42, verbose=0)
    }

    tree_results = {}
    reg_leaderboard = []
    clf_leaderboard = []

    for key, name, desc in TREE_ALGO_NAMES:
        print(f"Fitting {name}...")
        reg_model = tree_regressors[key]
        clf_model = tree_classifiers[key]

        # Fit Regressor
        reg_model.fit(X_train_imputed, y_lin_train)
        y_reg_train_pred = reg_model.predict(X_train_imputed)
        y_reg_test_pred = reg_model.predict(X_test_imputed)

        r2_tr = round(r2_score(y_lin_train, y_reg_train_pred), 4)
        r2_te = round(r2_score(y_lin_test, y_reg_test_pred), 4)
        mse_val = round(mean_squared_error(y_lin_test, y_reg_test_pred), 4)
        rmse_val = round(np.sqrt(mse_val), 4)

        # Regressor Feature Importances
        if hasattr(reg_model, "feature_importances_"):
            reg_importances = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLS, reg_model.feature_importances_)}
        else:
            reg_importances = {col: 0.2 for col in FEATURE_COLS}

        # Fit Classifier
        if key in ["xgboost", "lightgbm"]:
            # Uses 0-indexed targets
            clf_model.fit(X_train_imputed, y_log_train_0)
            y_clf_train_pred = clf_model.predict(X_train_imputed) + 1
            y_clf_test_pred = clf_model.predict(X_test_imputed) + 1
        elif key == "catboost":
            clf_model.fit(X_train_imputed, y_log_train)
            y_clf_train_pred = np.array(clf_model.predict(X_train_imputed)).flatten()
            y_clf_test_pred = np.array(clf_model.predict(X_test_imputed)).flatten()
        else:
            clf_model.fit(X_train_imputed, y_log_train)
            y_clf_train_pred = clf_model.predict(X_train_imputed)
            y_clf_test_pred = clf_model.predict(X_test_imputed)

        acc_tr = round(accuracy_score(y_log_train, y_clf_train_pred), 4)
        acc_te = round(accuracy_score(y_log_test, y_clf_test_pred), 4)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_log_test, y_clf_test_pred, average="weighted", zero_division=0)
        c_prec, c_rec, c_f1, c_supp = precision_recall_fscore_support(y_log_test, y_clf_test_pred, labels=classes, zero_division=0)

        clf_class_metrics = {}
        for i, c in enumerate(classes):
            clf_class_metrics[str(c)] = {
                "precision": round(float(c_prec[i]), 4),
                "recall": round(float(c_rec[i]), 4),
                "f1": round(float(c_f1[i]), 4),
                "support": int(c_supp[i])
            }

        conf_mat = confusion_matrix(y_log_test, y_clf_test_pred, labels=classes)

        # Classifier Feature Importances
        if hasattr(clf_model, "feature_importances_"):
            clf_importances = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLS, clf_model.feature_importances_)}
        else:
            clf_importances = {col: 0.2 for col in FEATURE_COLS}

        # Structure per-algorithm summary
        tree_results[key] = {
            "name": name,
            "description": desc,
            "linear": {
                "r2_train": r2_tr,
                "r2_test": r2_te,
                "mse": mse_val,
                "rmse": rmse_val,
                "feature_importances": reg_importances,
                "tree_structure": serialize_tree(reg_model, FEATURE_COLS) if key == "decision_tree" else None
            },
            "logistic": {
                "accuracy_train": acc_tr,
                "accuracy_test": acc_te,
                "precision_weighted": round(prec_w, 4),
                "recall_weighted": round(rec_w, 4),
                "f1_weighted": round(f1_w, 4),
                "class_metrics": clf_class_metrics,
                "feature_importances": clf_importances,
                "confusion_matrix": conf_mat.tolist(),
                "tree_structure": serialize_tree(clf_model, FEATURE_COLS) if key == "decision_tree" else None
            }
        }

        reg_leaderboard.append({
            "key": key,
            "name": name,
            "r2_test": r2_te,
            "mse": mse_val,
            "rmse": rmse_val
        })

        clf_leaderboard.append({
            "key": key,
            "name": name,
            "accuracy_test": acc_te,
            "precision_weighted": round(prec_w, 4),
            "recall_weighted": round(rec_w, 4),
            "f1_weighted": round(f1_w, 4)
        })

    # Sort leaderboards
    reg_leaderboard = sorted(reg_leaderboard, key=lambda x: x["r2_test"], reverse=True)
    clf_leaderboard = sorted(clf_leaderboard, key=lambda x: x["accuracy_test"], reverse=True)

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
        # 7 Trees Regressors & Classifiers
        "linear_decision_tree": tree_regressors["decision_tree"],
        "logistic_decision_tree": tree_classifiers["decision_tree"],
        "linear_random_forest": tree_regressors["random_forest"],
        "logistic_random_forest": tree_classifiers["random_forest"],
        "linear_adaboost": tree_regressors["adaboost"],
        "logistic_adaboost": tree_classifiers["adaboost"],
        "linear_gradient_boosting": tree_regressors["gradient_boosting"],
        "logistic_gradient_boosting": tree_classifiers["gradient_boosting"],
        "linear_xgboost": tree_regressors["xgboost"],
        "logistic_xgboost": tree_classifiers["xgboost"],
        "linear_lightgbm": tree_regressors["lightgbm"],
        "logistic_lightgbm": tree_classifiers["lightgbm"],
        "linear_catboost": tree_regressors["catboost"],
        "logistic_catboost": tree_classifiers["catboost"]
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
        # Decision Tree (backward compatibility)
        "decision_tree": tree_results["decision_tree"],
        # Complete 7 Tree-Based Algorithms Suite
        "tree_models": {
            "algorithms": tree_results,
            "leaderboard_regression": reg_leaderboard,
            "leaderboard_classification": clf_leaderboard,
            "algo_list": [{"key": k, "name": n, "description": d} for k, n, d in TREE_ALGO_NAMES]
        }
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("All models trained and cached successfully!")
    return results

def predict_sample(features_dict, category="regression", model_type="none"):
    """
    Performs real-time predictions using the selected dashboard category and model subtype.
    category: 'regression', 'regularization', 'tree', 'decision_tree'
    model_type:
      - regression/regularization: 'none' / 'ols', 'l2' / 'ridge', 'l1' / 'lasso', 'elasticnet'
      - tree: 'decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'
    """
    tree_keys = [k for k, _, _ in TREE_ALGO_NAMES]
    norm_type = str(model_type).lower().strip()

    alias_map = {
        "none": "ols",
        "ols": "ols",
        "l2": "ridge",
        "ridge": "ridge",
        "l1": "lasso",
        "lasso": "lasso",
        "elasticnet": "elasticnet"
    }

    if category in ["tree", "decision_tree"] or norm_type in tree_keys:
        chosen_tree = norm_type if norm_type in tree_keys else "decision_tree"
        linear_key = f"linear_{chosen_tree}"
        logistic_key = f"logistic_{chosen_tree}"
        is_tree_model = True
    else:
        sub = alias_map.get(norm_type, "ols")
        linear_key = f"linear_{sub}"
        logistic_key = f"logistic_{sub}"
        is_tree_model = False

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
    
    # Trees use raw imputed features; Linear/Logistic use scaled features
    if is_tree_model:
        input_processed = pd.DataFrame(imputer.transform(input_data), columns=FEATURE_COLS)
    else:
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=FEATURE_COLS)
        input_processed = pd.DataFrame(scaler.transform(input_imputed), columns=FEATURE_COLS)

    # Predict Linear/Distance
    pred_distance = float(lin_model.predict(input_processed)[0])

    # Predict Logistic/Severity & Probabilities
    if is_tree_model and model_type in ["xgboost", "lightgbm"]:
        # 0-indexed models
        pred_raw = int(log_model.predict(input_processed)[0])
        pred_severity = pred_raw + 1
    elif is_tree_model and model_type == "catboost":
        pred_val = log_model.predict(input_processed)
        pred_severity = int(np.array(pred_val).flatten()[0])
    else:
        pred_severity = int(log_model.predict(input_processed)[0])

    # Probabilities
    probs_dict = {}
    if hasattr(log_model, "predict_proba"):
        try:
            pred_probs = log_model.predict_proba(input_processed)[0]
            if hasattr(log_model, "classes_"):
                classes = [int(c) for c in log_model.classes_]
                if model_type in ["xgboost", "lightgbm"] and min(classes) == 0:
                    classes = [c + 1 for c in classes]
            else:
                classes = [1, 2, 3, 4]
            probs_dict = {str(c): round(float(p), 4) for c, p in zip(classes, pred_probs)}
        except Exception:
            probs_dict = {str(pred_severity): 1.0}
    else:
        probs_dict = {str(pred_severity): 1.0}

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
    print("Executing train_models()...")
    res = train_models(force_run=True)
    print("Training finished successfully!")
    print("Linear OLS R2:", res["regression"]["linear"]["r2_test"])
    print("Tree Leaderboard (Regression):", [f"{x['name']}: R2={x['r2_test']}" for x in res["tree_models"]["leaderboard_regression"][:3]])
    print("Tree Leaderboard (Classification):", [f"{x['name']}: Acc={x['accuracy_test']}" for x in res["tree_models"]["leaderboard_classification"][:3]])
