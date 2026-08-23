import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

file_path = r"C:\Users\yeswanth sai\PycharmProjects\PythonProject9\data\US_Accidents_Sample_100k.csv"
df = pd.read_csv(file_path)

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
