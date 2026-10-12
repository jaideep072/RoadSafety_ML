"""
Regression model evaluation metrics module.
Calculates R², Adjusted R², MAE, MSE, RMSE, and Explained Variance.
"""

from typing import Dict, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    explained_variance_score
)


@dataclass
class RegressionMetrics:
    r2: float
    adj_r2: float
    mae: float
    mse: float
    rmse: float
    explained_variance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_regression_model(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    n_features: int = 1
) -> RegressionMetrics:
    """
    Computes genuine, data-backed regression performance metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    n_samples = len(y_true_arr)
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Adjusted R2 computation
    if n_samples > n_features + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        adj_r2 = r2

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    ev = float(explained_variance_score(y_true_arr, y_pred_arr))

    return RegressionMetrics(
        r2=round(r2 * 100.0, 2),
        adj_r2=round(adj_r2 * 100.0, 2),
        mae=round(mae, 4),
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        explained_variance=round(ev, 4)
    )
