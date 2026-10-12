"""
Decision Tree and Ensemble Models configuration for RoadSafety_ML.
Configures exactly 6 Tree and Ensemble Algorithms matching the reference project:
1. ID3 Decision Tree (Entropy / Information Gain)
2. Random Forest (Bagging & Random Subspaces)
3. AdaBoost (Adaptive Boosting)
4. Gradient Boosting (Stage-Wise Additive Residual Boosting)
5. XGBoost (Extreme Gradient Boosting)
6. LightGBM (Histogram-based Leaf-Wise Gradient Boosting)
"""

from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


ALGO_CONFIG: Dict[str, Dict[str, Any]] = {
    "id3": {
        "name": "1. ID3 Decision Tree",
        "dropdown_name": "ID3 Decision Tree (Entropy / Information Gain)",
        "category": "Single Decision Tree (Information Gain)",
        "description": "Quinlan's ID3 algorithm calculates Shannon Entropy and Information Gain reduction to determine optimal decision nodes for accident severity classification.",
        "classifier": lambda: DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=20, random_state=42),
        "regressor": lambda: DecisionTreeRegressor(criterion="friedman_mse", max_depth=6, min_samples_split=20, random_state=42),
        "file_prefix": "id3",
        "hyperparams": {"criterion": "entropy (Information Gain)", "max_depth": "6", "min_samples_split": "20", "random_state": "42"}
    },
    "rf": {
        "name": "2. Random Forest",
        "dropdown_name": "Random Forest",
        "category": "Bagging & Random Subspaces",
        "description": "Ensemble of decorrelated decision trees aggregating variance across bootstrap subsamples and random feature subsets.",
        "classifier": lambda: RandomForestClassifier(n_estimators=50, max_depth=8, max_features="sqrt", random_state=42, n_jobs=-1),
        "regressor": lambda: RandomForestRegressor(n_estimators=50, max_depth=8, max_features="sqrt", random_state=42, n_jobs=-1),
        "file_prefix": "rf",
        "hyperparams": {"n_estimators": "50", "max_depth": "8", "max_features": "sqrt", "random_state": "42"}
    },
    "adaboost": {
        "name": "3. AdaBoost",
        "dropdown_name": "AdaBoost",
        "category": "Adaptive Boosting",
        "description": "Adaptive Boosting trains decision stumps sequentially, iteratively increasing sample weights on previously misclassified incident instances.",
        "classifier": lambda: AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42),
        "regressor": lambda: AdaBoostRegressor(n_estimators=50, learning_rate=0.8, random_state=42),
        "file_prefix": "adaboost",
        "hyperparams": {"n_estimators": "50", "learning_rate": "0.8", "random_state": "42"}
    },
    "gbm": {
        "name": "4. Gradient Boosting (GBM)",
        "dropdown_name": "Gradient Boosting (GBM)",
        "category": "Stage-Wise Additive Boosting",
        "description": "Gradient Boosting builds additive sequential trees fitted to minimize the pseudo-residuals of the prior ensemble stage.",
        "classifier": lambda: GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
        "regressor": lambda: GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
        "file_prefix": "gb",
        "hyperparams": {"n_estimators": "50", "max_depth": "4", "learning_rate": "0.1", "random_state": "42"}
    },
    "xgboost": {
        "name": "5. XGBoost",
        "dropdown_name": "XGBoost",
        "category": "Extreme Gradient Boosting",
        "description": "Extreme Gradient Boosting applies second-order Taylor expansions and built-in L1/L2 leaf penalty regularization.",
        "classifier": lambda: XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"),
        "regressor": lambda: XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
        "file_prefix": "xgb",
        "hyperparams": {"n_estimators": "50", "max_depth": "4", "learning_rate": "0.1", "eval_metric": "logloss"}
    },
    "lightgbm": {
        "name": "6. LightGBM",
        "dropdown_name": "LightGBM",
        "category": "Histogram-based Gradient Boosting (Leaf-Wise)",
        "description": "LightGBM utilizes histogram-based continuous feature binning and leaf-wise depth growth for high speed and accuracy.",
        "classifier": lambda: LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1),
        "regressor": lambda: LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1),
        "file_prefix": "lgbm",
        "hyperparams": {"n_estimators": "50", "max_depth": "4", "learning_rate": "0.1", "verbose": "-1"}
    }
}


def train_ensemble_model(algo_key: str = "id3", is_classification: bool = True):
    """Instantiates a classifier or regressor for the selected tree algorithm."""
    if algo_key not in ALGO_CONFIG:
        algo_key = "id3"
    config = ALGO_CONFIG[algo_key]
    factory = config["classifier"] if is_classification else config["regressor"]
    return factory(), config
