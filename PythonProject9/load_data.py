import os
import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\jdeep\Downloads\PythonProject9 (1)\PythonProject9\data\US_Accidents_March23.csv"

def load_data(path: str = DATA_PATH, nrows: int = 100000) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, nrows=nrows)
    return df

def get_data_summary() -> dict:
    df = load_data()
    
    # Replace NaN with None so it is valid JSON (null)
    preview_df = df.head(10).replace({np.nan: None})
    
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": preview_df.to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    print(get_data_summary())
