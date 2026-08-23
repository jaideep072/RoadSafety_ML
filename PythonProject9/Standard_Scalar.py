import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\yeswanth sai\PycharmProjects\PythonProject9\data\US_Accidents_Sample_100k.csv"
df = pd.read_csv(file_path)

num_cols = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("\n--- StandardScaler Standardization ---")
print("Columns to Standardize:", num_cols)

print("\nTraining Data BEFORE Standardization:")
print(train_df[num_cols].head())
print("\nTesting Data BEFORE Standardization:")
print(test_df[num_cols].head())

scaler = StandardScaler()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print("\nTraining Data AFTER Standardization (Mean=0, Std=1):")
print(train_df[num_cols].head())
print("\nTesting Data AFTER Standardization (Mean=0, Std=1):")
print(test_df[num_cols].head())

print("\nStandardScaler standardization completed successfully!")
