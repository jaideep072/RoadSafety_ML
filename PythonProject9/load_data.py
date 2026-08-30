import os
import pandas as pd
import numpy as np
import random

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "US_Accidents_March23.csv")

def generate_dummy_data(path: str, nrows: int = 10000):
    print(f"Dataset not found at {path}. Automatically generating a portable synthetic dataset of {nrows} rows...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Columns definition matching the US Accidents dataset schema
    data = {
        "ID": [f"A-{i}" for i in range(1, nrows + 1)],
        "Source": ["Source1" if i % 2 == 0 else "Source2" for i in range(nrows)],
        "Severity": np.random.choice([1, 2, 3, 4], size=nrows, p=[0.1, 0.6, 0.2, 0.1]),
        "Start_Time": [f"2023-01-01 {random.randint(0,23):02d}:{random.randint(0,59):02d}:00" for _ in range(nrows)],
        "End_Time": [f"2023-01-01 {random.randint(0,23):02d}:{random.randint(0,59):02d}:00" for _ in range(nrows)],
        "Start_Lat": np.random.uniform(24.396308, 49.384358, size=nrows),
        "Start_Lng": np.random.uniform(-124.848974, -66.885444, size=nrows),
        "End_Lat": [np.nan] * nrows,
        "End_Lng": [np.nan] * nrows,
        "Distance(mi)": np.random.exponential(scale=0.5, size=nrows),
        "Description": [f"Accident preview description {i}" for i in range(nrows)],
        "Number": np.random.choice([np.nan, 100.0, 500.0, 1000.0], size=nrows),
        "Street": [f"Main St {i}" for i in range(nrows)],
        "Side": ["R" if i % 3 == 0 else "L" for i in range(nrows)],
        "City": ["Charlotte" if i % 5 == 0 else "Houston" for i in range(nrows)],
        "County": ["Harris" if i % 2 == 0 else "Mecklenburg" for i in range(nrows)],
        "State": ["TX" if i % 2 == 0 else "NC" for i in range(nrows)],
        "Zipcode": [f"{random.randint(10000, 99999)}" for _ in range(nrows)],
        "Country": ["US"] * nrows,
        "Timezone": ["US/Central" if i % 2 == 0 else "US/Eastern" for i in range(nrows)],
        "Airport_Code": [f"K{random.choice(['CLT', 'IAH', 'ORD', 'LAX'])}" for _ in range(nrows)],
        "Weather_Timestamp": [f"2023-01-01 {random.randint(0,23):02d}:{random.randint(0,59):02d}:00" for _ in range(nrows)],
        "Temperature(F)": np.random.normal(loc=65.0, scale=15.0, size=nrows),
        "Wind_Chill(F)": np.random.normal(loc=60.0, scale=15.0, size=nrows),
        "Humidity(%)": np.clip(np.random.normal(loc=60.0, scale=20.0, size=nrows), 0, 100),
        "Pressure(in)": np.random.normal(loc=29.92, scale=0.5, size=nrows),
        "Visibility(mi)": np.clip(np.random.normal(loc=9.0, scale=2.0, size=nrows), 0, 10),
        "Wind_Direction": np.random.choice(["Calm", "N", "S", "E", "W", "NE", "NW", "SE", "SW", "VAR", "NNE", "NNW", "SSE", "SSW", "ENE", "WNW", "ESE", "WSW"], size=nrows),
        "Wind_Speed(mph)": np.clip(np.random.normal(loc=8.0, scale=5.0, size=nrows), 0, 50),
        "Precipitation(in)": np.clip(np.random.normal(loc=0.01, scale=0.1, size=nrows), 0, 2),
        "Weather_Condition": np.random.choice(["Clear", "Fair", "Mostly Cloudy", "Partly Cloudy", "Cloudy", "Overcast", "Light Rain", "Rain", "Heavy Rain", "Fog"], size=nrows),
        "Amenity": np.random.choice([True, False], size=nrows, p=[0.05, 0.95]),
        "Bump": np.random.choice([True, False], size=nrows, p=[0.01, 0.99]),
        "Crossing": np.random.choice([True, False], size=nrows, p=[0.08, 0.92]),
        "Give_Way": np.random.choice([True, False], size=nrows, p=[0.02, 0.98]),
        "Junction": np.random.choice([True, False], size=nrows, p=[0.1, 0.9]),
        "No_Exit": np.random.choice([True, False], size=nrows, p=[0.01, 0.99]),
        "Railway": np.random.choice([True, False], size=nrows, p=[0.02, 0.98]),
        "Roundabout": np.random.choice([True, False], size=nrows, p=[0.01, 0.99]),
        "Station": np.random.choice([True, False], size=nrows, p=[0.03, 0.97]),
        "Stop": np.random.choice([True, False], size=nrows, p=[0.04, 0.96]),
        "Traffic_Calming": np.random.choice([True, False], size=nrows, p=[0.01, 0.99]),
        "Traffic_Signal": np.random.choice([True, False], size=nrows, p=[0.12, 0.88]),
        "Turning_Loop": np.random.choice([True, False], size=nrows, p=[0.00, 1.00]),
        "Sunrise_Sunset": np.random.choice(["Day", "Night"], size=nrows, p=[0.65, 0.35]),
        "Civil_Twilight": np.random.choice(["Day", "Night"], size=nrows, p=[0.67, 0.33]),
        "Nautical_Twilight": np.random.choice(["Day", "Night"], size=nrows, p=[0.7, 0.3]),
        "Astronomical_Twilight": np.random.choice(["Day", "Night"], size=nrows, p=[0.72, 0.28])
    }
    
    # Introduce random NaNs
    for col in ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)", "Humidity(%)"]:
        mask = np.random.rand(nrows) < 0.05
        data[col] = [np.nan if m else v for m, v in zip(mask, data[col])]
        
    for col in ["Wind_Direction", "Sunrise_Sunset", "Civil_Twilight", "Weather_Condition"]:
        mask = np.random.rand(nrows) < 0.02
        data[col] = [None if m else v for m, v in zip(mask, data[col])]
        
    for col in ["Zipcode", "Timezone", "Airport_Code"]:
        mask = np.random.rand(nrows) < 0.01
        data[col] = [None if m else v for m, v in zip(mask, data[col])]
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset generated successfully at {path} ({nrows} rows).")

def load_data(path: str = DATA_PATH, nrows: int = 100000) -> pd.DataFrame:
    if not os.path.exists(path):
        generate_dummy_data(path, nrows=10000)
    df = pd.read_csv(path, nrows=nrows)
    return df

def get_data_summary() -> dict:
    df = load_data()
    
    # Replace NaN with None so it is valid JSON (null)
    preview_df = df.head(10).replace({np.nan: None})
    
    total_cells = int(df.shape[0] * df.shape[1])
    total_missing = int(df.isnull().sum().sum())
    completeness_pct = round(100.0 - (total_missing / total_cells * 100.0), 2) if total_cells > 0 else 100.0
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    bool_cols = list(df.select_dtypes(include=['bool']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime', 'datetime64']).columns)

    column_details = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing_cnt = int(df[col].isnull().sum())
        missing_pct = round((missing_cnt / df.shape[0]) * 100.0, 2)
        unique_cnt = int(df[col].nunique(dropna=True))
        
        column_details.append({
            "name": col,
            "dtype": dtype_str,
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "unique_count": unique_cnt,
            "is_numeric": col in numeric_cols,
            "is_categorical": col in categorical_cols,
            "is_bool": col in bool_cols,
        })
    
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "formatted_rows": f"{df.shape[0]:,}",
        "formatted_cols": f"{df.shape[1]:,}",
        "total_cells": total_cells,
        "total_missing": total_missing,
        "completeness_pct": completeness_pct,
        "memory_mb": memory_mb,
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "bool_cols_count": len(bool_cols),
        "other_cols_count": len(df.columns) - (len(numeric_cols) + len(categorical_cols) + len(bool_cols)),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "column_details": column_details,
        "preview": preview_df.to_dict("records"),
        "filename": os.path.basename(DATA_PATH),
    }
    return summary

if __name__ == "__main__":
    print(get_data_summary())
