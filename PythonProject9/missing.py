import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (Project 14 - pointing to lightweight 100k sample)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
df = pd.read_csv(file_path, nrows=100000)

print("Dataset shape:", df.shape)

# Split dataset into training and testing data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Severity"]
)

# 1. Per-column missing percentage in training data
missing = train_df.isna().mean() * 100
print("\nMissing values percentage in training data (sorted):")
print(missing[missing > 0].sort_values(ascending=False).round(2))

# 2. Cross-tabulate missing values of Temperature(F) against Sunrise_Sunset
print("\nTemperature(F) missing percentage grouped by Sunrise_Sunset:")
missing_by_sunset = train_df.groupby("Sunrise_Sunset")["Temperature(F)"].apply(
    lambda x: x.isna().mean() * 100
)
print(missing_by_sunset.round(2))
