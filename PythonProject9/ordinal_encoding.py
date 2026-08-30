import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
df = pd.read_csv(file_path, nrows=100000)

print("Dataset shape:", df.shape)

ordinal_cols = ["Sunrise_Sunset", "Civil_Twilight"]
for col in ordinal_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Severity"]
)

twilight_order = ["Night", "Day"]

ord_enc = OrdinalEncoder(
    categories=[
        twilight_order,
        twilight_order
    ]
)

train_df[["Sunrise_Sunset_enc", "Civil_Twilight_enc"]] = ord_enc.fit_transform(train_df[ordinal_cols])
test_df[["Sunrise_Sunset_enc", "Civil_Twilight_enc"]] = ord_enc.transform(test_df[ordinal_cols])

print("\nEncoded training columns (First 5 rows):")
print(train_df[["Sunrise_Sunset", "Sunrise_Sunset_enc", "Civil_Twilight", "Civil_Twilight_enc"]].head())

print("\nEncoded testing columns (First 5 rows):")
print(test_df[["Sunrise_Sunset", "Sunrise_Sunset_enc", "Civil_Twilight", "Civil_Twilight_enc"]].head())
