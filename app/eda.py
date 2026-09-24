"""
Exploratory Data Analysis (EDA) module for RoadSafety_ML.
Generates 14+ publication-quality statistical charts, correlation heatmaps,
temporal patterns, weather risk distributions, and geographic scatter plots.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from app.config import CHARTS_DIR, EDA_CACHE_PATH
from app.data.loader import load_dataset

logger = logging.getLogger(__name__)

# Styling configuration
sns.set_theme(style="whitegrid")


def _chart_path(filename: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHARTS_DIR / filename


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")


def run_eda(force_run: bool = False, nrows: int = 100000) -> Dict[str, Any]:
    """
    Executes the 14+ chart Exploratory Data Analysis suite and caches output.
    """
    if not force_run and EDA_CACHE_PATH.exists():
        try:
            with open(EDA_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_results = json.load(f)

            charts_list = cached_results.get("charts", [])
            all_exist = True
            for c in charts_list:
                fname = c["filename"] if isinstance(c, dict) else c
                if not (_chart_path(fname)).exists():
                    all_exist = False
                    break

            if all_exist and len(charts_list) > 0:
                logger.info("Returning cached EDA results.")
                return cached_results
        except Exception as e:
            logger.warning(f"Failed to read EDA cache, generating fresh: {e}")

    logger.info("Executing comprehensive EDA chart generation...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_type="original", nrows=nrows)

    if "Start_Time" in data.columns:
        data["Start_Time"] = pd.to_datetime(data["Start_Time"], errors="coerce")

    charts = []

    # 1. Missing Values
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        top_missing = missing_df.head(15)
        sns.barplot(x=top_missing.index, y=top_missing["missing_pct"], palette="Reds_r")
        plt.title("Top 15 Columns by Missing Value Percentage", fontsize=12, fontweight="bold")
        plt.xlabel("Features", fontsize=10)
        plt.ylabel("Missing Percentage (%)", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        _save("missing_values.png")
        charts.append({
            "filename": "missing_values.png",
            "title": "Missing Values Analysis",
            "description": "Identifies features with missingness to inform imputation and feature filtering."
        })

    # 2. Severity Distribution
    if "Severity" in data.columns:
        plt.figure(figsize=(7, 4.5), dpi=100)
        ax = sns.countplot(x="Severity", data=data, palette="Blues_d")
        plt.title("Accident Severity Distribution (Tiers 1 to 4)", fontsize=12, fontweight="bold")
        plt.xlabel("Severity Level", fontsize=10)
        plt.ylabel("Accident Count", fontsize=10)
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{int(height):,}", (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=8.5, xytext=(0, 3), textcoords="offset points")
        _save("severity_distribution.png")
        charts.append({
            "filename": "severity_distribution.png",
            "title": "Accident Severity Distribution",
            "description": "Class balance across the four severity tiers."
        })

    # 3. Accidents by Hour of Day
    if "Start_Time" in data.columns and data["Start_Time"].dt.hour.notna().any():
        plt.figure(figsize=(10, 4.5), dpi=100)
        data["Hour"] = data["Start_Time"].dt.hour
        sns.countplot(x="Hour", data=data, palette="Blues_r")
        plt.title("Hourly Accident Frequency (Rush Hour Spikes)", fontsize=12, fontweight="bold")
        plt.xlabel("Hour of Day (0 to 23)", fontsize=10)
        plt.ylabel("Accident Count", fontsize=10)
        _save("accidents_by_hour.png")
        charts.append({
            "filename": "accidents_by_hour.png",
            "title": "Accidents by Hour of Day",
            "description": "Highlights morning (7-9 AM) and evening (4-6 PM) rush hour crash concentrations."
        })

    # 4. Accidents by Day of Week
    if "Start_Time" in data.columns and data["Start_Time"].dt.day_name().notna().any():
        plt.figure(figsize=(8, 4.5), dpi=100)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        data["DayOfWeek"] = data["Start_Time"].dt.day_name()
        sns.countplot(x="DayOfWeek", data=data, order=day_order, palette="Blues_d")
        plt.title("Accident Frequency by Day of Week", fontsize=12, fontweight="bold")
        plt.xlabel("Day of Week", fontsize=10)
        plt.ylabel("Accident Count", fontsize=10)
        plt.xticks(rotation=20)
        _save("accidents_by_dayofweek.png")
        charts.append({
            "filename": "accidents_by_dayofweek.png",
            "title": "Accidents by Day of Week",
            "description": "Demonstrates weekday commuter traffic volume vs. weekend recreation patterns."
        })

    # 5. Accidents by Month
    if "Start_Time" in data.columns and data["Start_Time"].dt.month.notna().any():
        plt.figure(figsize=(10, 4.5), dpi=100)
        data["Month"] = data["Start_Time"].dt.month
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax = sns.countplot(x="Month", data=data, palette="Blues_r")
        plt.title("Seasonal Accident Frequency by Month", fontsize=12, fontweight="bold")
        plt.xlabel("Month", fontsize=10)
        plt.ylabel("Accident Count", fontsize=10)
        plt.xticks(ticks=range(0, 12), labels=month_names[:len(ax.get_xticks())])
        _save("accidents_by_month.png")
        charts.append({
            "filename": "accidents_by_month.png",
            "title": "Accidents by Month",
            "description": "Illustrates seasonal shifts and winter weather impact on road incidents."
        })

    # 6. Top 10 States by Accident Count
    if "State" in data.columns:
        plt.figure(figsize=(9, 4.5), dpi=100)
        top_states = data["State"].value_counts().head(10)
        sns.barplot(x=top_states.index, y=top_states.values, palette="Blues_r")
        plt.title("Top 10 US States with Highest Reported Incident Volumes", fontsize=12, fontweight="bold")
        plt.xlabel("State Code", fontsize=10)
        plt.ylabel("Total Incidents", fontsize=10)
        _save("top_states.png")
        charts.append({
            "filename": "top_states.png",
            "title": "Top 10 States by Incident Volume",
            "description": "Geographical state distribution reflecting highway density and population size."
        })

    # 7. Weather Conditions Distribution
    if "Weather_Condition" in data.columns:
        plt.figure(figsize=(10, 5), dpi=100)
        top_weather = data["Weather_Condition"].value_counts().head(10)
        sns.barplot(x=top_weather.values, y=top_weather.index, palette="Blues_d")
        plt.title("Top 10 Weather Conditions During Incidents", fontsize=12, fontweight="bold")
        plt.xlabel("Incident Count", fontsize=10)
        plt.ylabel("Weather Condition", fontsize=10)
        _save("weather_conditions.png")
        charts.append({
            "filename": "weather_conditions.png",
            "title": "Weather Conditions",
            "description": "Breakdown of prevailing meteorological conditions at time of collision."
        })

    # 8. Temperature by Severity
    if "Temperature(F)" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(8, 4.5), dpi=100)
        sns.boxplot(x="Severity", y="Temperature(F)", data=data, palette="Blues", showfliers=False)
        plt.title("Ambient Temperature Distribution by Accident Severity", fontsize=12, fontweight="bold")
        plt.xlabel("Severity Level", fontsize=10)
        plt.ylabel("Temperature (°F)", fontsize=10)
        _save("temperature_by_severity.png")
        charts.append({
            "filename": "temperature_by_severity.png",
            "title": "Temperature vs. Severity",
            "description": "Temperature interquartile spreads across severity tiers."
        })

    # 9. Visibility by Severity
    if "Visibility(mi)" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(8, 4.5), dpi=100)
        sns.boxplot(x="Severity", y="Visibility(mi)", data=data, palette="Blues", showfliers=False)
        plt.title("Atmospheric Visibility by Accident Severity", fontsize=12, fontweight="bold")
        plt.xlabel("Severity Level", fontsize=10)
        plt.ylabel("Visibility (miles)", fontsize=10)
        _save("visibility_by_severity.png")
        charts.append({
            "filename": "visibility_by_severity.png",
            "title": "Visibility vs. Severity",
            "description": "Assesses correlation between low visibility conditions and crash severity."
        })

    # 10. Day vs. Night Severity Breakdown
    if "Sunrise_Sunset" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(8, 4.5), dpi=100)
        sns.countplot(x="Severity", hue="Sunrise_Sunset", data=data.dropna(subset=["Sunrise_Sunset"]), palette="Blues")
        plt.title("Accident Severity: Day vs. Night Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Severity Level", fontsize=10)
        plt.ylabel("Incident Count", fontsize=10)
        plt.legend(title="Time of Day")
        _save("day_night_vs_severity.png")
        charts.append({
            "filename": "day_night_vs_severity.png",
            "title": "Day vs. Night vs. Severity",
            "description": "Compares diurnal illumination differences on accident severity."
        })

    # 11. Road Infrastructure Feature Impacts
    infra_cols = ["Crossing", "Junction", "Traffic_Signal", "Railway", "Station", "Stop"]
    present_infra = [col for col in infra_cols if col in data.columns]
    if present_infra:
        plt.figure(figsize=(9, 4.5), dpi=100)
        infra_counts = {col: int(data[col].sum()) for col in present_infra if pd.api.types.is_bool_dtype(data[col]) or pd.api.types.is_numeric_dtype(data[col])}
        if infra_counts:
            sns.barplot(x=list(infra_counts.keys()), y=list(infra_counts.values()), palette="Blues_d")
            plt.title("Accident Frequencies Near Road Infrastructure Elements", fontsize=12, fontweight="bold")
            plt.xlabel("Road Infrastructure Type", fontsize=10)
            plt.ylabel("Number of Incidents", fontsize=10)
            _save("infrastructure_impact.png")
            charts.append({
                "filename": "infrastructure_impact.png",
                "title": "Road Infrastructure Impact",
                "description": "Incident counts near signals, pedestrian crossings, and highway junctions."
            })

    # 12. Correlation Heatmap
    numeric_df = data.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 2:
        plt.figure(figsize=(10, 7.5), dpi=100)
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True, annot_kws={"size": 7.5})
        plt.title("Pearson Correlation Heatmap of Numerical Features", fontsize=12, fontweight="bold")
        _save("correlation_heatmap.png")
        charts.append({
            "filename": "correlation_heatmap.png",
            "title": "Correlation Heatmap",
            "description": "Evaluates linear dependencies and multicollinearity across sensor variables."
        })

    # 13. Geographic Scatter Map
    if "Start_Lat" in data.columns and "Start_Lng" in data.columns:
        plt.figure(figsize=(10, 6), dpi=100)
        geo_sample = data.dropna(subset=["Start_Lat", "Start_Lng"])
        if len(geo_sample) > 5000:
            geo_sample = geo_sample.sample(n=5000, random_state=42)
        plt.scatter(geo_sample["Start_Lng"], geo_sample["Start_Lat"], color="#0284c7", alpha=0.35, s=10)
        plt.title("Geographical Incident Scatter Map (US)", fontsize=12, fontweight="bold")
        plt.xlabel("Longitude (°W)", fontsize=10)
        plt.ylabel("Latitude (°N)", fontsize=10)
        _save("accident_heatmap.png")
        charts.append({
            "filename": "accident_heatmap.png",
            "title": "Geographical Incident Scatter",
            "description": "Spatial distribution across the continental United States highway network."
        })

    summary_stats = {
        "n_rows": len(data),
        "n_cols": data.shape[1],
        "duplicate_count": int(data.duplicated().sum()),
        "charts": charts
    }

    EDA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)
    logger.info(f"Saved EDA cache: {EDA_CACHE_PATH}")

    return summary_stats
