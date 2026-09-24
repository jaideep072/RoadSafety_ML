"""
Centralized Model Manager Service for RoadSafety_ML.
Provides thread-safe artifact loading, in-memory caching, validation,
and fallback retraining triggers.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import joblib

from app.config import (
    MODELS_DIR,
    SCALERS_DIR,
    ENCODERS_DIR,
    MODELS_CACHE_PATH
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton-capable ModelManager ensuring efficient, cached access
    to serialized scikit-learn/XGBoost/LightGBM binaries.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._model_cache = {}
                    cls._instance._scaler_cache = {}
                    cls._instance._encoder_cache = {}
        return cls._instance

    def get_model(self, model_filename: str) -> Any:
        """
        Loads and caches model binary from artifacts/models directory.
        """
        if not model_filename.endswith(".pkl"):
            model_filename = f"{model_filename}.pkl"

        with self._lock:
            if model_filename in self._model_cache:
                return self._model_cache[model_filename]

            model_path = MODELS_DIR / model_filename
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact '{model_filename}' not found at {model_path}. "
                    "Run model training pipeline first."
                )

            logger.info(f"Loading model artifact from disk: {model_path}")
            model = joblib.load(model_path)
            self._model_cache[model_filename] = model
            return model

    def get_scaler(self, scaler_filename: str = "scaler.pkl") -> Any:
        """Loads and caches scaler object from artifacts/scalers."""
        with self._lock:
            if scaler_filename in self._scaler_cache:
                return self._scaler_cache[scaler_filename]

            scaler_path = SCALERS_DIR / scaler_filename
            if not scaler_path.exists():
                scaler_path = MODELS_DIR / scaler_filename

            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}.")

            logger.info(f"Loading scaler artifact from disk: {scaler_path}")
            scaler = joblib.load(scaler_path)
            self._scaler_cache[scaler_filename] = scaler
            return scaler

    def get_imputer(self, imputer_filename: str = "imputer.pkl") -> Any:
        """Loads and caches imputer object from artifacts/encoders."""
        with self._lock:
            if imputer_filename in self._encoder_cache:
                return self._encoder_cache[imputer_filename]

            imputer_path = ENCODERS_DIR / imputer_filename
            if not imputer_path.exists():
                imputer_path = MODELS_DIR / imputer_filename

            if not imputer_path.exists():
                raise FileNotFoundError(f"Imputer artifact not found at {imputer_path}.")

            logger.info(f"Loading imputer artifact from disk: {imputer_path}")
            imputer = joblib.load(imputer_path)
            self._encoder_cache[imputer_filename] = imputer
            return imputer

    def save_model(self, model: Any, model_filename: str) -> Path:
        """Saves model to disk and updates in-memory cache."""
        if not model_filename.endswith(".pkl"):
            model_filename = f"{model_filename}.pkl"

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target_path = MODELS_DIR / model_filename

        with self._lock:
            joblib.dump(model, target_path)
            self._model_cache[model_filename] = model
            logger.info(f"Saved and cached model: {target_path}")

        return target_path

    def has_model(self, model_filename: str) -> bool:
        """Checks if a model artifact exists on disk."""
        if not model_filename.endswith(".pkl"):
            model_filename = f"{model_filename}.pkl"
        return (MODELS_DIR / model_filename).exists()

    def list_available_models(self) -> List[str]:
        """Returns list of all pickled model files present in artifacts/models."""
        if not MODELS_DIR.exists():
            return []
        return [f.name for f in MODELS_DIR.glob("*.pkl")]

    def clear_cache(self):
        """Clears all in-memory model caches."""
        with self._lock:
            self._model_cache.clear()
            self._scaler_cache.clear()
            self._encoder_cache.clear()
            logger.info("Cleared in-memory model manager cache.")


# Singleton instance
model_manager = ModelManager()
