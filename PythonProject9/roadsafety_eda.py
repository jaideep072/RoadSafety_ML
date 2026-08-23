import os
import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for web application

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_data import load_data

# Set styling
sns.set_theme(style="whitegrid")

# Charts directory configuration
CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)
CACHE_FILE = os.path.join(CHARTS_DIR, "results_cache.json")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")


def run_eda(force_run: bool = False) -> dict:
    # 0. CHECK CACHE
    if not force_run and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_results = json.load(f)

            charts_list = cached_results.get("charts", [])
            all_exist = True
            for c in charts_list:
                fname = c["filename"] if isinstance(c, dict) else c
                if not os.path.exists(os.path.join(CHARTS_DIR, fname)):
                    all_exist = False
                    break

            is_new_format = len(charts_list) > 0 and isinstance(charts_list[0], dict)

            if all_exist and is_new_format:
                print("\n========== RETURNING CACHED EDA RESULTS INSTANTLY ==========")
                return cached_results
        except Exception as e:
            print("Cache read failed, running EDA fresh:", e)

    print("\n========== EDA STARTED ==========")

    # 1. LOAD DATA
    data = load_data()
    print("=" * 80)
    print("1. Data loaded")
    print("=" * 80)
    print("shape:", data.shape)
    print("\nFirst 5 rows of data:\n", data.head())

    charts = []

    # Convert Start_Time to datetime for temporal analyses
    if "Start_Time" in data.columns:
        # Some rows might have nan or inconsistent formats; convert safely
        data["Start_Time"] = pd.to_datetime(data["Start_Time"], errors="coerce")

    # 2. BASIC INFO / STRUCTURE
    print("\n" + "=" * 80)
    print("2. BASIC INFO")
    print("=" * 80)
    data.info()
    print("\nColumns dtypes:\n", data.dtypes)
    print("\nDescribe (numeric):\n", data.describe())
    try:
        print("\nDescribe (categorical):\n", data.describe(include="object"))
    except ValueError:
        print("\nDescribe (categorical): No categorical columns to describe.")

    # 3. MISSING VALUES
    print("\n" + "=" * 80)
    print("1. MISSING VALUES")
    print("=" * 80)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)
    print(missing_df)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        # Select top 15 missing columns to keep the chart clean
        top_missing_df = missing_df.head(15)
        sns.barplot(x=top_missing_df.index, y=top_missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Percentage of missing values")
        plt.title("Missing values by column (Top 15)")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # 4. DUPLICATES
    print("\n" + "=" * 80)
    print("2. Duplicate Rows")
    print("=" * 80)
    duplicate_count = int(data.duplicated().sum())
    print("Duplicate rows:", duplicate_count)

    # 5. TARGET VARIABLE - SEVERITY DISTRIBUTION
    print("\n" + "=" * 80)
    print("3. TARGET VARIABLE - SEVERITY")
    print("=" * 80)
    target_counts = {}
    if "Severity" in data.columns:
        target_counts = data["Severity"].value_counts().to_dict()
        print(data["Severity"].value_counts())

        plt.figure(dpi=125)
        sns.countplot(x="Severity", data=data)
        plt.xlabel("Severity (1 = Low, 4 = High)")
        plt.ylabel("Count")
        plt.title("Accident Severity Distribution")
        _save("severity_distribution.png")
        charts.append("severity_distribution.png")

    # 6. HOURLY TREND
    print("\n" + "=" * 80)
    print("4. HOURLY TREND ANALYSIS")
    print("=" * 80)
    if "Start_Time" in data.columns:
        data["Hour"] = data["Start_Time"].dt.hour
        hourly_counts = data["Hour"].dropna().value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        plt.plot(hourly_counts.index, hourly_counts.values, marker="o", color="crimson")
        plt.title("Accidents Hourly Trend")
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Accident Count")
        plt.xticks(range(0, 24))
        _save("accidents_by_hour.png")
        charts.append("accidents_by_hour.png")

    # 7. DAY OF WEEK TREND
    print("\n" + "=" * 80)
    print("5. DAY OF WEEK ANALYSIS")
    print("=" * 80)
    if "Start_Time" in data.columns:
        data["DayOfWeek"] = data["Start_Time"].dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        plt.figure(figsize=(10, 5))
        sns.countplot(x="DayOfWeek", data=data, order=days_order, palette="viridis")
        plt.title("Accidents by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Count")
        plt.xticks(rotation=15)
        _save("accidents_by_dayofweek.png")
        charts.append("accidents_by_dayofweek.png")

    # 8. MONTHLY TREND (SEASONALITY)
    print("\n" + "=" * 80)
    print("6. MONTHLY SEASONAL TREND")
    print("=" * 80)
    if "Start_Time" in data.columns:
        data["Month"] = data["Start_Time"].dt.month_name()
        months_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        # Only order by months present in dataset
        present_months = [m for m in months_order if m in data["Month"].dropna().unique()]

        plt.figure(figsize=(10, 5))
        sns.countplot(x="Month", data=data, order=present_months, palette="magma")
        plt.title("Accidents by Month")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.xticks(rotation=30)
        _save("accidents_by_month.png")
        charts.append("accidents_by_month.png")

    # 9. WEATHER CONDITIONS ANALYSIS
    print("\n" + "=" * 80)
    print("7. WEATHER CONDITION ANALYSIS")
    print("=" * 80)
    if "Weather_Condition" in data.columns:
        weather_counts = data["Weather_Condition"].value_counts().head(10)
        print("Top 10 weather conditions during accidents:\n", weather_counts)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=weather_counts.values, y=weather_counts.index, palette="Blues_r")
        plt.title("Top 10 Weather Conditions during Accidents")
        plt.xlabel("Accident Count")
        plt.ylabel("Weather Condition")
        _save("weather_conditions.png")
        charts.append("weather_conditions.png")

    # 10. STATE-WISE ACCIDENT COUNTS
    print("\n" + "=" * 80)
    print("8. STATE-WISE ACCIDENT DISTRIBUTION")
    print("=" * 80)
    if "State" in data.columns:
        state_counts = data["State"].value_counts().head(10)
        print("Top 10 states with highest accident counts:\n", state_counts)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=state_counts.index, y=state_counts.values, palette="rocket")
        plt.title("Top 10 States by Accident Count")
        plt.xlabel("State")
        plt.ylabel("Accident Count")
        _save("top_states.png")
        charts.append("top_states.png")

    # 11. TEMPERATURE VS SEVERITY
    print("\n" + "=" * 80)
    print("9. TEMPERATURE VS SEVERITY")
    print("=" * 80)
    if "Temperature(F)" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Severity", y="Temperature(F)", data=data, palette="Set2")
        plt.title("Temperature Distribution across Severity Levels")
        plt.xlabel("Severity")
        plt.ylabel("Temperature (F)")
        _save("temperature_by_severity.png")
        charts.append("temperature_by_severity.png")

    # 12. VISIBILITY VS SEVERITY
    print("\n" + "=" * 80)
    print("10. VISIBILITY VS SEVERITY")
    print("=" * 80)
    if "Visibility(mi)" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Severity", y="Visibility(mi)", data=data, palette="Set2")
        plt.title("Visibility Distribution across Severity Levels")
        plt.xlabel("Severity")
        plt.ylabel("Visibility (miles)")
        _save("visibility_by_severity.png")
        charts.append("visibility_by_severity.png")

    # 13. INFRASTRUCTURE IMPACT
    print("\n" + "=" * 80)
    print("11. INFRASTRUCTURE FACTOR ANALYSIS")
    print("=" * 80)
    infra_cols = ["Crossing", "Junction", "Railway", "Station", "Stop", "Traffic_Signal"]
    infra_cols = [c for c in infra_cols if c in data.columns]

    if infra_cols:
        # Sum Boolean values to get count of True values
        infra_counts = {col: int(data[col].sum()) for col in infra_cols}
        infra_df = pd.DataFrame(list(infra_counts.items()), columns=["Feature", "Count"]).sort_values(by="Count",
                                                                                                      ascending=False)
        print("Accidents occurred near infrastructure elements:\n", infra_df)

        plt.figure(figsize=(10, 5))
        sns.barplot(x="Feature", y="Count", data=infra_df, palette="coolwarm")
        plt.title("Accidents near Key Road Infrastructure Elements")
        plt.xlabel("Infrastructure Element")
        plt.ylabel("Accident Count")
        _save("infrastructure_impact.png")
        charts.append("infrastructure_impact.png")

    # 14. DAY VS NIGHT SEVERITY
    print("\n" + "=" * 80)
    print("12. DAY/NIGHT SEVERITY ANALYSIS")
    print("=" * 80)
    if "Sunrise_Sunset" in data.columns and "Severity" in data.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x="Sunrise_Sunset", hue="Severity", data=data, palette="Set1")
        plt.title("Accident Severity: Day vs Night")
        plt.xlabel("Time of Day")
        plt.ylabel("Count")
        plt.legend(title="Severity")
        _save("day_night_vs_severity.png")
        charts.append("day_night_vs_severity.png")

    # 15. ACCIDENT HEATMAP (GEOGRAPHIC DISPERSAL)
    print("\n" + "=" * 80)
    print("13. ACCIDENT GEOGRAPHIC HEATMAP")
    print("=" * 80)
    if "Start_Lat" in data.columns and "Start_Lng" in data.columns:
        # Downsample to maximum 5,000 points to plot quickly
        heatmap_df = data[["Start_Lat", "Start_Lng", "Severity"]].dropna()
        if len(heatmap_df) > 5000:
            heatmap_df = heatmap_df.sample(n=5000, random_state=42)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="Start_Lng", y="Start_Lat", hue="Severity",
            palette="plasma", data=heatmap_df, alpha=0.5, s=15
        )
        plt.title("Geographic Clustering of Accidents (Sampled)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Severity")
        _save("accident_heatmap.png")
        charts.append("accident_heatmap.png")

    # 16. CORRELATION HEATMAP
    print("\n" + "=" * 80)
    print("14. CORRELATION HEATMAP")
    print("=" * 80)
    # Focus on meaningful environmental and geographical numerical columns
    potential_cols = ['Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 
                      'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
                      'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    cols_to_use = [col for col in potential_cols if col in data.columns]
    
    if len(cols_to_use) > 1:
        corr_data = data[cols_to_use]
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        plt.figure(figsize=(12, 10))
        # Mask to show only bottom triangle (optional but makes it cleaner, though full matrix is fine, let's keep full for simplicity)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, 
                    linewidths=0.5, annot_kws={"size": 10}, square=True)
        
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.title("Correlation Heatmap of Numerical Features", fontsize=14, pad=20)
        _save("correlation_heatmap.png")
        charts.append("correlation_heatmap.png")

    print("\n========== EDA COMPLETED ==========")
    print("Charts generated:", len(charts))

    # Map raw chart names to human-readable titles in the returned results
    chart_title_map = {
        "missing_values.png": "Missing Values by Column (Top 15)",
        "severity_distribution.png": "Accident Severity Distribution",
        "accidents_by_hour.png": "Hourly Accident Trend (Peak Hours)",
        "accidents_by_dayofweek.png": "Weekly Accident Trend (Day of Week)",
        "accidents_by_month.png": "Accidents by Month (Seasonal Trend)",
        "weather_conditions.png": "Top 10 Weather Conditions during Accidents",
        "top_states.png": "Top 10 States by Accident Count",
        "temperature_by_severity.png": "Impact of Temperature on Accident Severity",
        "visibility_by_severity.png": "Impact of Visibility on Accident Severity",
        "infrastructure_impact.png": "Accidents Near Key Infrastructure Elements",
        "day_night_vs_severity.png": "Accident Severity: Day vs. Night",
        "accident_heatmap.png": "Geographic Clustering of Accidents (Sampled)",
        "correlation_heatmap.png": "Correlation Heatmap of Numerical Features"
    }

    formatted_charts = []
    for fname in charts:
        formatted_charts.append({
            "filename": fname,
            "title": chart_title_map.get(fname, fname.replace(".png", "").replace("_", " ").title())
        })

    results = {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicate_count,
        "missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0},
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "charts": formatted_charts,
    }

    # Save to cache
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f)
        print("EDA results cached successfully.")
    except Exception as e:
        print("Failed to save EDA results to cache:", e)

    return results


if __name__ == "__main__":
    results = run_eda()
    print("\n================================================")
    print("EDA RESULTS PREVIEW")
    print("================================================")
    print("Rows:", results["n_rows"])
    print("Columns:", results["n_cols"])
    print("Duplicate rows:", results["duplicate_count"])
    print("Missing values count:", results["missing"])
    print("Severity counts:", results["target_counts"])
    print("Charts generated:")
    for chart in results["charts"]:
        print(" -", chart)
