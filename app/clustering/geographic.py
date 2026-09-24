"""
Geographical Accident Concentration and Spatial Density Analysis for RoadSafety_ML.
Computes latitude/longitude coordinate spatial distributions, density concentrations,
and cluster zones using strictly neutral, empirical terminology.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.config import CHARTS_DIR
from app.data.loader import load_dataset
from app.data.validators import validate_geographic_coordinates


def run_geographic_analysis(
    df: Optional[pd.DataFrame] = None,
    lat_col: str = "Start_Lat",
    lng_col: str = "Start_Lng",
    sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Analyzes geographic accident distributions:
    1. Ensures valid latitude/longitude coordinate data.
    2. Computes spatial density metrics.
    3. Generates 2D geographic scatter and density visualization.
    4. Provides empirical observations with neutral phrasing.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []

    if df is None or lat_col not in df.columns or lng_col not in df.columns:
        df = load_dataset(dataset_type="original")

    clean_geo = validate_geographic_coordinates(df, lat_col=lat_col, lng_col=lng_col)

    is_sampled = False
    if len(clean_geo) > sample_size:
        clean_geo = clean_geo.sample(n=sample_size, random_state=random_state)
        is_sampled = True
        warnings.append(f"Geographical map rendered on a representative sample of {sample_size:,} records.")

    # Calculate spatial bounding metrics
    lat_min, lat_max = float(clean_geo[lat_col].min()), float(clean_geo[lat_col].max())
    lng_min, lng_max = float(clean_geo[lng_col].min()), float(clean_geo[lng_col].max())
    lat_center, lng_center = float(clean_geo[lat_col].mean()), float(clean_geo[lng_col].mean())

    chart_filename = "geographic_accident_density.png"
    plt.figure(figsize=(9, 5.5), dpi=120)

    if "Severity" in clean_geo.columns:
        sns.scatterplot(
            data=clean_geo,
            x=lng_col,
            y=lat_col,
            hue="Severity",
            palette="viridis",
            alpha=0.4,
            s=18,
            legend="full"
        )
    else:
        plt.scatter(
            clean_geo[lng_col],
            clean_geo[lat_col],
            color="#0284c7",
            alpha=0.35,
            s=15,
            edgecolors="none"
        )

    plt.title("Geographical Incident Density & Observed Concentration Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Longitude (°W)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / chart_filename, bbox_inches="tight")
    plt.close()

    return {
        "total_points": len(clean_geo),
        "is_sampled": is_sampled,
        "lat_range": [round(lat_min, 4), round(lat_max, 4)],
        "lng_range": [round(lng_min, 4), round(lng_max, 4)],
        "center": [round(lat_center, 4), round(lng_center, 4)],
        "chart_filename": chart_filename,
        "disclaimer": "Concentrations reflect historical reported incidents within the dataset observation window.",
        "warnings": warnings
    }
