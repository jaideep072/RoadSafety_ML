import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
df = pd.read_csv(file_path, nrows=100000)

num_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("\n--- Outlier Detection and Capping using IQR Method ---")
print("Target Columns:", num_cols)

for col in num_cols:
    print(f"\nProcessing Column: {col}")
    
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    
    train_outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)].shape[0]
    test_outliers = test_df[(test_df[col] < lower_bound) | (test_df[col] > upper_bound)].shape[0]
    print(f"  Outliers in Train Set: {train_outliers} / {train_df.shape[0]} ({train_outliers / train_df.shape[0] * 100:.2f}%)")
    print(f"  Outliers in Test Set: {test_outliers} / {test_df.shape[0]} ({test_outliers / test_df.shape[0] * 100:.2f}%)")
    
    print("\n  Stats BEFORE Capping:")
    print("    Train Min:", train_df[col].min(), "| Max:", train_df[col].max())
    print("    Test Min:", test_df[col].min(), "| Max:", test_df[col].max())
    
    train_df[col] = train_df[col].clip(lower_bound, upper_bound)
    test_df[col] = test_df[col].clip(lower_bound, upper_bound)
    
    print("  Stats AFTER Capping:")
    print("    Train Min:", train_df[col].min(), "| Max:", train_df[col].max())
    print("    Test Min:", test_df[col].min(), "| Max:", test_df[col].max())

print("\nOutlier fixing completed successfully!")
