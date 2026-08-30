import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "US_Accidents_March23.csv")
df = pd.read_csv(file_path, nrows=100000)

print("Dataset shape:", df.shape)

nominal_col = ["Wind_Direction"]

for col in nominal_col:
    df[col] = df[col].fillna(df[col].mode()[0])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Severity"]
)

print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

train_ohe = ohe.fit_transform(train_df[nominal_col])
test_ohe = ohe.transform(test_df[nominal_col])

ohe_cols = ohe.get_feature_names_out(nominal_col)

train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_cols, index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_cols, index=test_df.index)

print("\nNumber of original categorical columns:", len(nominal_col))
print("Number of generated One-Hot columns:", len(ohe_cols))

print("\nFirst 5 rows of encoded training data:")
print(train_ohe_df.head())

print("\nFirst 5 rows of encoded testing data:")
print(test_ohe_df.head())
