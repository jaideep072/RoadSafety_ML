"""
Classification model evaluation metrics module.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC score.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: list
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_classification_model(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None
) -> ClassificationMetrics:
    """
    Computes genuine, data-backed classification performance metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = round(float(accuracy_score(y_true_arr, y_pred_arr)) * 100.0, 2)
    prec = round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)) * 100.0, 2)
    rec = round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)) * 100.0, 2)
    f1 = round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)) * 100.0, 2)

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    roc_auc = None
    if y_prob is not None:
        try:
            # If 2D proba array passed, take positive class
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2:
                y_prob_arr = y_prob_arr[:, 1]
            roc_auc = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
        except Exception:
            roc_auc = None

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm.tolist(),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn)
    )
