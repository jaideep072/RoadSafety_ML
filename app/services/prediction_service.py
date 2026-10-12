"""
Prediction Service for RoadSafety_ML.
Handles input sanitation, schema alignment, feature transformations,
and model inference for both REST API and server-rendered HTML views.
"""

from typing import Dict, Any, Optional
import pandas as pd

from app.config import (
    SUPERVISED_REGRESSION_FEATURES,
    SUPERVISED_CLASSIFICATION_FEATURES
)
from app.services.model_manager import model_manager
from app.models.linear_regression import predict_linear_severity
from app.models.logistic_regression import predict_logistic_severity
from app.models.decision_tree import predict_tree_severity


class PredictionService:
    """Orchestrates predictions across linear, logistic, and ensemble models."""

    def predict_regression(
        self,
        model_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predicts accident delay / impact distance."""
        filename = f"{model_name}.pkl" if not model_name.endswith(".pkl") else model_name
        model = model_manager.get_model(filename)
        prediction = predict_linear_severity(model, input_data, SUPERVISED_REGRESSION_FEATURES)
        
        return {
            "model": model_name,
            "prediction": prediction,
            "unit": "miles",
            "features_used": SUPERVISED_REGRESSION_FEATURES
        }

    def predict_classification(
        self,
        model_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predicts binary accident severity tier (Severe vs Moderate/Minor)."""
        filename = f"{model_name}.pkl" if not model_name.endswith(".pkl") else model_name
        model = model_manager.get_model(filename)
        res = predict_logistic_severity(model, input_data, SUPERVISED_CLASSIFICATION_FEATURES)
        if not res:
            res = predict_tree_severity(model, input_data, SUPERVISED_CLASSIFICATION_FEATURES)

        return {
            "model": model_name,
            "result": res,
            "features_used": SUPERVISED_CLASSIFICATION_FEATURES
        }


prediction_service = PredictionService()
