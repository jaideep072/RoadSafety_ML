import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error

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
    "Traffic_Signal"
]
TARGET = "Distance(mi)"

def train_and_evaluate_linear(reg_type="none", student_input=None):
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Preprocessed dataset not found at {DATA_PATH}. Run preprocessing first.")
        
    df = pd.read_csv(DATA_PATH)
    
    active_features = [f for f in FEATURES if f in df.columns]
    X = df[active_features].copy()
    y = df[TARGET].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if reg_type == "ridge":
        model = Ridge(alpha=1.0)
        model_name = "Ridge Regression (L2 Penalty)"
    elif reg_type == "lasso":
        model = Lasso(alpha=0.01)
        model_name = "Lasso Regression (L1 Penalty)"
    else:
        model = LinearRegression()
        model_name = "Standard Linear Regression (Ordinary Least Squares)"
        reg_type = "none"
        
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    test_r2 = round(r2_score(y_test, y_test_pred) * 100, 2)
    train_r2 = round(r2_score(y_train, y_train_pred) * 100, 2)
    rmse = round(root_mean_squared_error(y_test, y_test_pred), 4)
    mae = round(mean_absolute_error(y_test, y_test_pred), 4)
    mse = round(mean_squared_error(y_test, y_test_pred), 4)
    
    intercept = round(float(model.intercept_), 4)
    coefficients = []
    eq_terms = []
    for f, coef in zip(active_features, model.coef_):
        c_val = round(float(coef), 4)
        is_zero = abs(c_val) == 0.0
        coefficients.append({
            "feature": f,
            "weight": c_val,
            "is_zero": is_zero
        })
        sign = "+" if c_val >= 0 else "-"
        eq_terms.append(f"{sign} {abs(c_val)}*({f})")
        
    equation_str = f"Distance(mi) = {intercept} " + " ".join(eq_terms)
    
    plot_actual_vs_pred = f"lr_actual_pred_{reg_type}.png"
    plot_residuals = f"lr_residuals_{reg_type}.png"
    
    sample_idx = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    plt.figure(figsize=(6, 4.5), dpi=120)
    plt.scatter(y_test.iloc[sample_idx], y_test_pred[sample_idx], alpha=0.4, color="#1e3a8a", edgecolors="none")
    min_val = min(y_test.iloc[sample_idx].min(), y_test_pred[sample_idx].min())
    max_val = max(y_test.iloc[sample_idx].max(), y_test_pred[sample_idx].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual Distance (miles)", fontsize=10)
    plt.ylabel("Predicted Distance (miles)", fontsize=10)
    plt.title(f"Actual vs Predicted Distance ({model_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, plot_actual_vs_pred))
    plt.close()
    
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6, 4.5), dpi=120)
    plt.hist(residuals, bins=40, color="#0284c7", edgecolor="#0369a1", alpha=0.7)
    plt.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual Error (Actual - Predicted)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title(f"Residuals Error Distribution ({model_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, plot_residuals))
    plt.close()
    
    prediction = None
    if student_input:
        try:
            row_vals = [float(student_input.get(f, 0.0)) for f in active_features]
            input_df = pd.DataFrame([row_vals], columns=active_features)
            pred_val = model.predict(input_df)[0]
            prediction = round(max(0.0, float(pred_val)), 3)
        except Exception:
            prediction = None
            
    return {
        "reg_type": reg_type,
        "model_name": model_name,
        "metrics": {
            "test_r2": test_r2,
            "train_r2": train_r2,
            "rmse": rmse,
            "mae": mae,
            "mse": mse
        },
        "intercept": intercept,
        "coefficients": coefficients,
        "equation": equation_str,
        "plots": {
            "actual_vs_pred": plot_actual_vs_pred,
            "residuals": plot_residuals
        },
        "prediction": prediction,
        "features": active_features
    }
