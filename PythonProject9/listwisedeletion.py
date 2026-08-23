import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset (Project 14 - pointing to 100k sample)
file_path = r"C:\Users\yeswanth sai\PycharmProjects\PythonProject9\data\US_Accidents_Sample_100k.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# Identify numerical features with missing values
num_cols = [
    "Temperature(F)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Pressure(in)",
    "Humidity(%)"
]
target_col = "Severity"

# Clean target and select columns
df_subset = df[num_cols + [target_col]].dropna(subset=[target_col])
X = df_subset[num_cols]
y = df_subset[target_col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n--- BEFORE DELETION ---")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")

# Apply Listwise Deletion on training split
train_complete = pd.concat([X_train, y_train], axis=1).dropna()
X_train_clean = train_complete[num_cols]
y_train_clean = train_complete[target_col]

# Apply Listwise Deletion on testing split
test_complete = pd.concat([X_test, y_test], axis=1).dropna()
X_test_clean = test_complete[num_cols]
y_test_clean = test_complete[target_col]

data_loss_pct = (1 - (len(X_train_clean) / len(X_train))) * 100

print(f"\n--- AFTER LISTWISE DELETION ---")
print(f"Clean training rows: {len(X_train_clean):,}")
print(f"Clean testing rows : {len(X_test_clean):,}")
print(f"Data discarded     : {data_loss_pct:.1f}%")

# Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train_clean, y_train_clean)

# Evaluate
accuracy = model.score(X_test_clean, y_test_clean) * 100
print(f"\nModel Accuracy on complete cases: {accuracy:.2f}%")
