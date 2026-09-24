"""
RoadSafety_ML Application Configuration Module.
Provides centralized path management, environment settings, and directory references.
Uses pathlib for cross-platform compatibility across Windows, Linux, and macOS.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"

# Static & Templates
STATIC_DIR = APP_DIR / "static"
CHARTS_DIR = STATIC_DIR / "charts"
TEMPLATES_DIR = APP_DIR / "templates"

# Caches
MODELS_CACHE_PATH = STATIC_DIR / "models_cache.json"
PREPROCESSING_CACHE_PATH = STATIC_DIR / "preprocessing_cache.json"
EDA_CACHE_PATH = CHARTS_DIR / "results_cache.json"

# Dataset Files
SAMPLE_DATA_FILENAME = "US_Accidents_Sample_100k.csv"
FULL_DATA_FILENAME = "US_Accidents_March23.csv"
PREPROCESSED_DATA_FILENAME = "roadsafety_preprocessed.csv"

RAW_SAMPLE_PATH = RAW_DATA_DIR / SAMPLE_DATA_FILENAME
RAW_FULL_PATH = RAW_DATA_DIR / FULL_DATA_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PREPROCESSED_DATA_FILENAME

# Features & Targets Configuration
DEFAULT_NUMERIC_FEATURES = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)"
]

SUPERVISED_REGRESSION_FEATURES = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Crossing",
    "Junction",
    "Traffic_Signal"
]

SUPERVISED_CLASSIFICATION_FEATURES = [
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

CLUSTERING_FEATURES = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)"
]

REGRESSION_TARGET = "Distance(mi)"
CLASSIFICATION_TARGET = "Severity_Severe"

# Flask Configuration
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "roadsafety-secure-key-2026")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    PORT = int(os.environ.get("PORT", 5000))
    HOST = os.environ.get("HOST", "127.0.0.1")
    JSON_SORT_KEYS = False

    # Ensure required directories exist on init
    @classmethod
    def init_app(cls, app=None):
        for directory in [
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            MODELS_DIR,
            SCALERS_DIR,
            ENCODERS_DIR,
            STATIC_DIR,
            CHARTS_DIR,
            STATIC_DIR / "css",
            STATIC_DIR / "js",
            STATIC_DIR / "images"
        ]:
            directory.mkdir(parents=True, exist_ok=True)
