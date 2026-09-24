"""
Application Services package for RoadSafety_ML.
Includes ModelManager and PredictionService.
"""

from app.services.model_manager import ModelManager, model_manager
from app.services.prediction_service import PredictionService, prediction_service

__all__ = [
    "ModelManager",
    "model_manager",
    "PredictionService",
    "prediction_service"
]
