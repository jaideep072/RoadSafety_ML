"""
Ensemble Algorithms configuration and builder for RoadSafety_ML.
Configures Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and LightGBM (7 Benchmark Models).
"""

from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


ALGO_CONFIG: Dict[str, Dict[str, Any]] = {
    "dt": {
        "name": "1. Decision Tree",
        "category": "Base Tree Learner",
        "description": "Decision Tree recursively splits feature space using Gini Impurity to partition accidents into severity classes.",
        "classifier": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
        "regressor": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
        "file_prefix": "dt"
    },
    "bagging": {
        "name": "2. Bagging Classifier",
        "category": "Bagging Ensemble",
        "description": "Bootstrap Aggregating trains multiple parallel trees on bootstrap data subsets and combines votes to reduce variance.",
        "classifier": lambda: BaggingClassifier(n_estimators=30, random_state=42, n_jobs=-1),
        "regressor": lambda: None,
        "file_prefix": "bagging"
    },
    "rf": {
        "name": "3. Random Forest",
        "category": "Bagging + Random Subspaces",
        "description": "Random Forest builds decorrelated decision trees by sampling random feature subsets at every split.",
        "classifier": lambda: RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        "regressor": lambda: RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        "file_prefix": "rf"
    },
    "adaboost": {
        "name": "4. AdaBoost",
        "category": "Sequential Boosting",
        "description": "Adaptive Boosting trains decision trees sequentially, giving higher sample weight to previously misclassified accident instances.",
        "classifier": lambda: AdaBoostClassifier(n_estimators=50, random_state=42),
        "regressor": lambda: AdaBoostRegressor(n_estimators=50, random_state=42),
        "file_prefix": "adaboost"
    },
    "gbm": {
        "name": "5. Gradient Boosting (GBM)",
        "category": "Residual Boosting",
        "description": "Gradient Boosting builds additive sequential trees fitted to minimize the pseudo-residuals of the prior ensemble stage.",
        "classifier": lambda: GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
        "regressor": lambda: GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
        "file_prefix": "gb"
    },
    "xgboost": {
        "name": "6. XGBoost",
        "category": "Regularized Gradient Boosting",
        "description": "Extreme Gradient Boosting applies second-order Taylor expansions and built-in L1/L2 leaf penalty regularization.",
        "classifier": lambda: XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric="logloss"),
        "regressor": lambda: XGBRegressor(n_estimators=50, max_depth=4, random_state=42),
        "file_prefix": "xgb"
    },
    "lightgbm": {
        "name": "7. LightGBM",
        "category": "Histogram-based Gradient Boosting",
        "description": "LightGBM utilizes histogram-based continuous feature binning and leaf-wise depth growth for high speed and accuracy.",
        "classifier": lambda: LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1),
        "regressor": lambda: LGBMRegressor(n_estimators=50, max_depth=4, random_state=42, verbose=-1),
        "file_prefix": "lgbm"
    }
}


def train_ensemble_model(algo_key: str = "dt", is_classification: bool = True):
    """Instantiates a classifier or regressor for the selected ensemble algorithm."""
    if algo_key not in ALGO_CONFIG:
        algo_key = "dt"
    config = ALGO_CONFIG[algo_key]
    factory = config["classifier"] if is_classification else config["regressor"]
    return factory(), config
