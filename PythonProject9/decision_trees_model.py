import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "roadsafety_preprocessed.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

FEATURES = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Crossing",
    "Junction",
    "Traffic_Signal",
    "Sunrise_Sunset_Day"
]
TARGET = "Severity_Severe"

ALGO_CONFIG = {
    "dt": {
        "name": "1. Decision Tree",
        "category": "Base Tree Learner",
        "description": "Decision Tree recursively splits feature space using Gini Impurity to partition accidents into severity classes.",
        "model": DecisionTreeClassifier(max_depth=6, random_state=42)
    },
    "bagging": {
        "name": "2. Bagging Classifier",
        "category": "Bagging Ensemble",
        "description": "Bootstrap Aggregating trains multiple parallel trees on bootstrap data subsets and combines votes to reduce variance.",
        "model": BaggingClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    },
    "rf": {
        "name": "3. Random Forest",
        "category": "Bagging + Random Subspaces",
        "description": "Random Forest builds decorrelated decision trees by sampling random feature subsets at every split.",
        "model": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    },
    "adaboost": {
        "name": "4. AdaBoost",
        "category": "Sequential Boosting",
        "description": "Adaptive Boosting trains decision trees sequentially, giving higher sample weight to previously misclassified accident instances.",
        "model": AdaBoostClassifier(n_estimators=50, random_state=42)
    },
    "gbm": {
        "name": "5. Gradient Boosting (GBM)",
        "category": "Residual Boosting",
        "description": "Gradient Boosting builds additive sequential trees fitted to minimize the pseudo-residuals of the prior ensemble stage.",
        "model": GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    },
    "xgboost": {
        "name": "6. XGBoost",
        "category": "Regularized Gradient Boosting",
        "description": "Extreme Gradient Boosting applies second-order Taylor expansions and built-in L1/L2 leaf penalty regularization.",
        "model": XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric="logloss")
    },
    "lightgbm": {
        "name": "7. LightGBM",
        "category": "Histogram-based Gradient Boosting",
        "description": "LightGBM utilizes histogram-based continuous feature binning and leaf-wise depth growth for high speed and accuracy.",
        "model": LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1)
    }
}

def train_and_evaluate_tree(algo_key="dt", student_input=None):
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    if algo_key not in ALGO_CONFIG:
        algo_key = "dt"
        
    config = ALGO_CONFIG[algo_key]
    model = config["model"]
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Preprocessed dataset not found at {DATA_PATH}. Run preprocessing first.")
        
    df = pd.read_csv(DATA_PATH)
    active_features = [f for f in FEATURES if f in df.columns]
    X = df[active_features].copy()
    y = df[TARGET].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    test_acc = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    train_acc = round(accuracy_score(y_train, y_train_pred) * 100, 2)
    prec = round(precision_score(y_test, y_test_pred, zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_test_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, y_test_pred, zero_division=0) * 100, 2)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Feature Importances
    feature_importances = []
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        for f, imp in zip(active_features, raw_imp):
            feature_importances.append({
                "feature": f,
                "importance": round(float(imp) * 100, 2)
            })
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        mean_imp = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        for f, imp in zip(active_features, mean_imp):
            feature_importances.append({
                "feature": f,
                "importance": round(float(imp) * 100, 2)
            })
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
        
    # Visualizations
    cm_plot = f"dt_cm_{algo_key}.png"
    plt.figure(figsize=(5, 4), dpi=130)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Moderate/Minor", "Severe"],
                yticklabels=["Moderate/Minor", "Severe"],
                annot_kws={"size": 13, "weight": "bold"})
    plt.xlabel("Predicted Severity", fontsize=10, fontweight="bold")
    plt.ylabel("Actual Severity", fontsize=10, fontweight="bold")
    plt.title(f"Confusion Matrix ({config['name']})", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, cm_plot))
    plt.close()
    
    imp_plot = f"dt_imp_{algo_key}.png"
    if feature_importances:
        top_imp = feature_importances[:8]
        plt.figure(figsize=(6, 4), dpi=120)
        names = [item["feature"] for item in reversed(top_imp)]
        vals = [item["importance"] for item in reversed(top_imp)]
        plt.barh(names, vals, color="#0284c7")
        plt.xlabel("Importance (%)", fontsize=10)
        plt.title(f"Feature Importances ({config['name']})", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, imp_plot))
        plt.close()
    else:
        imp_plot = None
        
    # Tree visualization: ONLY for single Base Decision Tree
    tree_plot = None
    if algo_key == "dt":
        tree_plot = "dt_tree_dt.png"
        fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
        plot_tree(
            model,
            feature_names=active_features,
            class_names=["Moderate/Minor", "Severe"],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=3,
            ax=ax
        )
        plt.title("Level-Wise Decision Tree Architecture (Levels 0 to 3) — Base Tree", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, tree_plot), bbox_inches="tight", facecolor="white")
        plt.close()
        
    # Live Predictor
    prediction = None
    if student_input:
        try:
            row_vals = [float(student_input.get(f, 0.0)) for f in active_features]
            input_df = pd.DataFrame([row_vals], columns=active_features)
            pred_class = int(model.predict(input_df)[0])
            try:
                prob = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)
            except Exception:
                prob = 85.0 if pred_class == 1 else 15.0
            prediction = {
                "class": "Severe Accident" if pred_class == 1 else "Minor / Moderate Accident",
                "is_severe": pred_class == 1,
                "probability": prob
            }
        except Exception:
            prediction = None
            
    return {
        "algo_key": algo_key,
        "algo_name": config["name"],
        "category": config["category"],
        "description": config["description"],
        "metrics": {
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "feature_importances": feature_importances,
        "plots": {
            "cm_plot": cm_plot,
            "imp_plot": imp_plot,
            "tree_plot": tree_plot
        },
        "prediction": prediction,
        "algo_list": [(k, v["name"]) for k, v in ALGO_CONFIG.items()],
        "features": active_features
    }
