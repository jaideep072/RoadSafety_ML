import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

def train_and_evaluate_logistic(reg_type="none", student_input=None):
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Preprocessed dataset not found at {DATA_PATH}. Run preprocessing first.")
        
    df = pd.read_csv(DATA_PATH)
    
    active_features = [f for f in FEATURES if f in df.columns]
    X = df[active_features].copy()
    y = df[TARGET].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if reg_type == "ridge":
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        model_name = "Logistic Regression with Ridge (L2 Penalty)"
    elif reg_type == "lasso":
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42)
        model_name = "Logistic Regression with Lasso (L1 Penalty)"
    else:
        model = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
        model_name = "Standard Logistic Regression (Log-Loss, No Regularization)"
        reg_type = "none"
        
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    test_acc = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    train_acc = round(accuracy_score(y_train, y_train_pred) * 100, 2)
    prec = round(precision_score(y_test, y_test_pred, zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_test_pred, zero_division=0) * 100, 2)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    intercept = round(float(model.intercept_[0]), 4)
    coefficients = []
    eq_terms = []
    for f, coef in zip(active_features, model.coef_[0]):
        c_val = round(float(coef), 4)
        odds_ratio = round(float(np.exp(c_val)), 3)
        is_zero = abs(c_val) == 0.0
        coefficients.append({
            "feature": f,
            "weight": c_val,
            "odds_ratio": odds_ratio,
            "is_zero": is_zero
        })
        sign = "+" if c_val >= 0 else "-"
        eq_terms.append(f"{sign} {abs(c_val)}*({f})")
        
    equation_str = f"z = {intercept} " + " ".join(eq_terms)
    
    cm_plot = f"log_cm_{reg_type}.png"
    plt.figure(figsize=(5, 4), dpi=130)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Moderate/Minor", "Severe"],
                yticklabels=["Moderate/Minor", "Severe"],
                annot_kws={"size": 13, "weight": "bold"})
    plt.xlabel("Predicted Severity", fontsize=10, fontweight="bold")
    plt.ylabel("Actual Severity", fontsize=10, fontweight="bold")
    plt.title(f"Confusion Matrix ({model_name})", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, cm_plot))
    plt.close()
    
    prediction = None
    if student_input:
        try:
            row_vals = [float(student_input.get(f, 0.0)) for f in active_features]
            input_df = pd.DataFrame([row_vals], columns=active_features)
            pred_class = int(model.predict(input_df)[0])
            prob = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)
            prediction = {
                "class": "Severe Accident" if pred_class == 1 else "Minor / Moderate Accident",
                "is_severe": pred_class == 1,
                "probability": prob
            }
        except Exception:
            prediction = None
            
    return {
        "reg_type": reg_type,
        "model_name": model_name,
        "metrics": {
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "precision": prec,
            "recall": rec,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "intercept": intercept,
        "coefficients": coefficients,
        "equation": equation_str,
        "plots": {
            "cm_plot": cm_plot
        },
        "prediction": prediction,
        "features": active_features
    }
