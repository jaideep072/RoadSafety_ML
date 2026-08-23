import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
file_path = r"C:\Users\yeswanth sai\PycharmProjects\PythonProject9\data\US_Accidents_Sample_100k.csv"
df = pd.read_csv(file_path)

# Features containing missing values
num_cols = [
    "Temperature(F)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Pressure(in)",
    "Humidity(%)"
]
target_col = "Severity"

df_subset = df[num_cols + [target_col]].dropna(subset=[target_col])
X = df_subset[num_cols]
y = df_subset[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# 1. Listwise Deletion
train_complete = pd.concat([X_train, y_train], axis=1).dropna()
X_train_d = train_complete[num_cols]
y_train_d = train_complete[target_col]

test_complete = pd.concat([X_test, y_test], axis=1).dropna()
X_test_d = test_complete[num_cols]
y_test_d = test_complete[target_col]

model_d = LogisticRegression(max_iter=500)
model_d.fit(X_train_d, y_train_d)
acc_d = model_d.score(X_test_d, y_test_d) * 100
results["Listwise Deletion"] = {
    "Train Rows": len(X_train_d),
    "Test Accuracy": f"{acc_d:.2f}%"
}

# 2. Median Imputation
median_imputer = SimpleImputer(strategy="median")
X_train_med = X_train.copy()
X_test_med = X_test.copy()
X_train_med[num_cols] = median_imputer.fit_transform(X_train[num_cols])
X_test_med[num_cols] = median_imputer.transform(X_test[num_cols])

model_med = LogisticRegression(max_iter=500)
model_med.fit(X_train_med, y_train)
acc_med = model_med.score(X_test_med, y_test) * 100
results["Median Imputation"] = {
    "Train Rows": len(X_train_med),
    "Test Accuracy": f"{acc_med:.2f}%"
}

# 3. KNN Imputation (k=5)
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = X_train.copy()
X_test_knn = X_test.copy()
X_train_knn[num_cols] = knn_imputer.fit_transform(X_train[num_cols])
X_test_knn[num_cols] = knn_imputer.transform(X_test[num_cols])

model_knn = LogisticRegression(max_iter=500)
model_knn.fit(X_train_knn, y_train)
acc_knn = model_knn.score(X_test_knn, y_test) * 100
results["KNN Imputation (k=5)"] = {
    "Train Rows": len(X_train_knn),
    "Test Accuracy": f"{acc_knn:.2f}%"
}

# 4. Median + Missing-Indicator Pipeline
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols)
], remainder="passthrough")

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500))
])
pipe.fit(X_train, y_train)
acc_ind = pipe.score(X_test, y_test) * 100
results["Median + Missing Indicator"] = {
    "Train Rows": len(X_train),
    "Test Accuracy": f"{acc_ind:.2f}%"
}

print("\n" + "="*60)
print(f"{'Missing Value Handling Strategy':<30} | {'Train Rows':<10} | {'Test Accuracy':<15}")
print("="*60)
for strategy, metrics in results.items():
    print(f"{strategy:<30} | {metrics['Train Rows']:<10} | {metrics['Test Accuracy']:<15}")
print("="*60)
