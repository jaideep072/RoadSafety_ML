# Developer & Contributor Guide

## Getting Started

### 1. Environment Setup

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/jaideep072/RoadSafety_ML.git
cd RoadSafety_ML

python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
python run.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Running Pipelines via CLI

### Run Preprocessing Pipeline:
```bash
python scripts/preprocess_data.py
```

### Train All ML Models:
```bash
python scripts/train_models.py --force
```

### Run Clustering Suite:
```bash
python scripts/run_clustering.py --algo all
```

---

## Automated Testing

Execute the complete test suite with pytest:

```bash
python -m pytest -v
```

To run individual test modules:
```bash
python -m pytest tests/test_models.py -v
python -m pytest tests/test_clustering.py -v
python -m pytest tests/test_preprocessing.py -v
```

---

## Coding Standards

- **Project-Relative Paths**: Always use `pathlib.Path` from `app.config`. Never use hardcoded absolute machine paths (`C:\Users\...`).
- **No Data Leakage**: Always fit imputers, scalers, and encoders strictly on the training partition (`X_train`) before transforming validation or test sets.
- **Unfabricated Metrics**: Never hardcode or invent accuracy, R², silhouette scores, or cluster counts. All values are calculated directly on actual dataset splits.
- **Thread Safety**: Access serialized models via `model_manager` to leverage thread-safe in-memory caching.
