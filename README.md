# 🚦 RoadSafety AI — Accident Severity Prediction & Risk Analysis Suite

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg?style=for-the-badge)](https://github.com/jaideep072/RoadSafety_ML)

**An end-to-end Machine Learning web application, analytics platform, and clustering engine designed to analyze, predict, and discover traffic accident patterns using real-world US accident records.**

[Explore Features](#-key-features) • [Model Benchmarks](#-model-benchmarks--leaderboard) • [Clustering Suite](#-unsupervised-clustering-suite) • [System Architecture](#-system-architecture) • [Quickstart](#-quickstart-guide) • [Project Structure](#-project-structure) • [REST API](#-rest-api-reference)

</div>

---

## 📌 Overview

**RoadSafety AI** is an industrial-grade machine learning and spatial intelligence platform built on top of extensive real-world traffic incident telemetry data. The system provides an interactive Web Studio delivering:

1. **Automated Data Ingestion & Sanitization** with live memory profiling and schema validation.
2. **Exploratory Data Analysis (EDA)** featuring 14+ statistical charts and correlation heatmaps.
3. **Leakage-Free Preprocessing Lab** (IQR outlier capping, imputation benchmarks, One-Hot & Ordinal Encoders, StandardScaler & MinMaxScaler).
26: 4. **Supervised Learning Suite**:
27:    - **Regression**: Ordinary Least Squares (OLS), Ridge ($L_2$), Lasso ($L_1$), ElasticNet ($L_1 + L_2$).
28:    - **Classification**: Logistic Regression (OLS, Ridge, Lasso, ElasticNet).
29:    - **Tree & Ensembles**: Base Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting (GBM), XGBoost, and LightGBM (7 Benchmark Models).
30: 5. **Unsupervised Clustering Suite**:
31:    - **K-Means Clustering** with automated Elbow Method optimization and centroid analysis.
32:    - **DBSCAN Density Clustering** with automated Epsilon ($\epsilon$) detection via sorted $k$-distance knee point detection.
33:    - **Agglomerative Hierarchical Clustering** with linkage comparisons (Ward, Complete, Average, Single) and sample tree dendrograms.
34:    - **Geographic Spatial Analysis** mapping observed historical incident density concentrations.
35: 6. **Real-Time Interactive Prediction Studio** equipped with 1-click presets (*Stormy Highway*, *Clear City Day*, *Freezing Fog*, *Rush Hour Gridlock*) and zero-flicker Dark/Light theme switching.
36: 
37: ---
38: 
39: ## ✨ Key Features
40: 
41: - 🌓 **Zero-Flicker Dark & Light Mode**: Seamless theme switching with instant `localStorage` persistence across all views and interactive tables.
42: - ⚡ **1-Click Scenario Presets**: Auto-fill complex meteorological and temporal conditions (*Stormy Highway*, *Clear City Day*, *Freezing Fog*, *Rush Hour Gridlock*) for rapid testing.
43: - 📊 **14+ High-Resolution EDA Visualizations**: Weather condition distributions, temperature-severity spreads, visibility vs. severity, state-by-state heatmaps, and temporal crash frequencies.
44: - 🧪 **Preprocessing Lab**: Side-by-side empirical comparisons between Listwise Deletion, Median Imputation, KNN Imputation ($k=5$), and Median + Missing Indicator, alongside interactive IQR outlier boundaries and feature scaling previews.
45: - 🤖 **10+ Serialized ML Classifiers & Regressors**: Pre-trained pipeline models ready for sub-millisecond inference.
46: - 📍 **Spatial & Density Clustering**: K-Means, DBSCAN with automated Knee eps inflection detection, Hierarchical Dendrograms, and Geographic bounding density maps.
47: - 🎯 **Dual Interface Support**: Full server-side rendered **Flask + Jinja2 + Modern CSS3** interface paired with an optional **React 18 + Vite** client dashboard.
48: - 🛡️ **Automated Pytest Suite**: 30+ comprehensive unit, integration, and route validation tests ensuring 100% test passing rate.
49: 
50: ---
51: 
52: ## 🏗️ System Architecture
53: 
54: ```mermaid
55: flowchart TD
56:     A[Raw Traffic Data / US Accidents] --> B[Data Loader & Ingestion Engine]
57:     B --> C[EDA Suite - 14+ Chart Modules]
58:     B --> D[Preprocessing Pipeline]
59: 
60:     subgraph Preprocessing_Layer [Data Preprocessing & Transformation]
61:         D --> D1[Missing Value Imputer\nMean / Median / Mode / KNN]
62:         D --> D2[IQR Outlier Capping\nUpper / Lower Bounds]
63:         D --> D3[Categorical Encoders\nOne-Hot, Ordinal, Target]
64:         D --> D4[Feature Scalers\nStandardScaler & MinMaxScaler]
65:     end
66: 
67:     Preprocessing_Layer --> E[Supervised Learning Engine]
68:     Preprocessing_Layer --> CL[Unsupervised Clustering Suite]
69: 
70:     subgraph Supervised_Suite [Supervised Machine Learning Suite]
71:         E --> M1[Linear Regression OLS / Ridge / Lasso / ElasticNet]
72:         E --> M2[Logistic Regression OLS / Ridge / Lasso / ElasticNet]
73:         E --> M3[Decision Tree Regressor & Classifier]
74:         E --> M4[Ensembles: Random Forest, AdaBoost, GBM, XGBoost, LightGBM]
75:     end
76: 
77:     subgraph Clustering_Suite [Unsupervised Clustering & Spatial Intelligence]
78:         CL --> C1[K-Means Clustering + Elbow Method]
79:         CL --> C2[DBSCAN + Automated Knee Epsilon Selection]
80:         CL --> C3[Hierarchical Clustering + Dendrograms]
81:         CL --> C4[Geographic Concentration & Density Mapping]
82:     end
83: 
84:     Supervised_Suite --> F[(Serialized Artifacts\n.pkl Binaries)]
85:     G --> H[Prediction Service]
86:     F --> G[Thread-Safe ModelManager Service]
87:     H & Clustering_Suite --> I[Flask Backend & REST API]
88:     I --> J[Interactive Web Studio UI & React Dashboard]
89: ```
90: 
91: ---
92: 
93: ## 📊 Model Benchmarks & Leaderboard
94: 
95: All metrics are calculated directly from empirical evaluations on test data partitions:
96: 
97: ### 1. Regression Models (Predicting Delay / Blockage Distance in Miles)
98: 
99: | Model Name | Regularization Strategy | Penalty Parameters | $R^2$ Score | RMSE (mi) | MAE (mi) | Serialization Target |
100: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
101: | **Linear Regression** | Ordinary Least Squares | None | Evaluated | Evaluated | Evaluated | `artifacts/models/linear_reg.pkl` |
102: | **Ridge Regression** | $L_2$ Norm Weight Decay | $\alpha = 1.0$ | Evaluated | Evaluated | Evaluated | `artifacts/models/linear_ridge.pkl` |
103: | **Lasso Regression** | $L_1$ Norm Sparsity | $\alpha = 0.01$ | Evaluated | Evaluated | Evaluated | `artifacts/models/linear_lasso.pkl` |
104: | **ElasticNet** | Convex Combination ($L_1 + L_2$) | $l_1 = 0.5, \alpha = 0.01$ | Evaluated | Evaluated | Evaluated | `artifacts/models/linear_elasticnet.pkl` |
105: | **Decision Tree Regressor**| Gini Split Partitioning | Max Depth: 8 | Evaluated | Evaluated | Evaluated | `artifacts/models/dt_regressor.pkl` |
106: | **Random Forest Regressor**| Bagging + Feature Sampling | 30 Estimators | Evaluated | Evaluated | Evaluated | `artifacts/models/rf_regressor.pkl` |
107: | **Gradient Boosting** | Additive Pseudo-Residuals | 30 Estimators | Evaluated | Evaluated | Evaluated | `artifacts/models/gb_regressor.pkl` |
108: | **XGBoost Regressor** | Regularized 2nd-Order Gradients | 30 Estimators | Evaluated | Evaluated | Evaluated | `artifacts/models/xgb_regressor.pkl` |
109: | **LightGBM Regressor** | Histogram Binning + Leaf-wise | 30 Estimators | Evaluated | Evaluated | Evaluated | `artifacts/models/lgbm_regressor.pkl` |
110: 
111: ### 2. Classification Models (Severe vs. Minor Incident Tier)
112: 
113: | Classifier Name | Strategy / Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
114: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
115: | **Logistic Regression** | Unregularized (Log-Loss) | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
116: | **Logistic (Ridge)** | $L_2$ Weight Penalty | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
117: | **Logistic (Lasso)** | $L_1$ Feature Selection | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
118: | **Logistic (ElasticNet)** | Dual $L_1 / L_2$ Regularization | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
119: | **Decision Tree** | Gini Impurity Splitting | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
120: | **Random Forest** | Bootstrap Aggregating | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
121: | **AdaBoost** | Sequential Adaptive Boosting | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
122: | **Gradient Boosting** | Residual Optimization | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
123: | **XGBoost** | Second-Order Gradient Boosting | Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |
124: | **LightGBM** | Histogram-based Gradient Boosting| Evaluated | Evaluated | Evaluated | Evaluated | Evaluated |

---

## 🔬 Unsupervised Clustering Suite

### 1. K-Means Clustering
- **Features**: Multi-dimensional spatial & meteorological inputs (`Start_Lat`, `Start_Lng`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`).
- **Elbow Method Curve**: Computes Inertia across $k \in [2, 8]$ to evaluate optimal clustering compactness.
- **Centroids Table**: Displays cluster center averages in original physical units (°F, %, in, mi).

### 2. DBSCAN & Automated Knee Detection
- **Density Clustering**: Detects arbitrary-shaped spatial accident clusters without requiring $k$ to be specified upfront.
- **Breakdown**: Segregates Core points, Border points, and Dispersed Noise outliers (`label = -1`).
- **Automated Knee Detection (`knee_for_dbscan.py`)**: Computes sorted $k$-nearest neighbor distances and detects the maximum perpendicular curvature point (Kneedle algorithm) to recommend the optimal $\epsilon$.

### 3. Hierarchical Agglomerative Clustering
- **Linkages Supported**: Ward (Minimum Variance), Complete, Average, and Single.
- **Dendrogram Visualizer**: Generates subtree merger diagrams on a clean representative sample.

### 4. Geographic Density Analysis
- **Spatial Bounding**: Maps historical accident concentration zones using strictly empirical, neutral terminology (*"higher observed accident density"*).

---

## 🚀 Quickstart Guide

### Prerequisites
- Python `3.10` or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/jaideep072/RoadSafety_ML.git
cd RoadSafety_ML
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python run.py
```

Open your browser and navigate to **`http://127.0.0.1:5000`** to access the RoadSafety Studio interface.

---

## 📂 Project Structure

```text
RoadSafety_ML/
├── README.md                           # Master documentation & benchmark showcase
├── requirements.txt                    # Audited production dependencies
├── .gitignore                          # Git version control ignore definitions
├── LICENSE                             # MIT License
├── run.py                              # Application entry point
│
├── app/                                # Main application package
│   ├── __init__.py                     # Flask factory & smart template url_for handler
│   ├── config.py                       # Centralized pathlib paths & application settings
│   ├── routes.py                       # Server-rendered HTML & JSON REST API blueprint
│   ├── eda.py                          # 14+ statistical chart generator & cache manager
│   │
│   ├── data/                           # Data loading and schema validation
│   │   ├── __init__.py
│   │   ├── loader.py                   # Dataset loader & memory downcasting
│   │   └── validators.py               # Schema, coordinate, and clustering validators
│   │
│   ├── preprocessing/                  # Data preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # End-to-end preprocessing orchestrator
│   │   ├── missing.py                  # Missing data diagnostics
│   │   ├── imputation.py               # Leakage-free imputation benchmarks
│   │   ├── outliers.py                 # IQR outlier detection & capping
│   │   ├── encoding.py                 # One-Hot, Ordinal, Target & Embedding encoders
│   │   └── scaling.py                  # StandardScaler & MinMaxScaler
│   │
│   ├── models/                         # Supervised Machine Learning
│   │   ├── __init__.py
│   │   ├── linear_regression.py        # OLS, Ridge, Lasso, ElasticNet
│   │   ├── logistic_regression.py      # Logistic Classifier (Unregularized/Ridge/Lasso/ElasticNet)
│   │   ├── decision_tree.py            # Base Decision Tree & Tree Visualizer
│   │   ├── ensembles.py                # Bagging, RF, AdaBoost, GBM, XGB, LGBM (7 Models)
│   │   └── pipeline.py                 # 10+ Model Training & serialization engine
│   │
│   ├── clustering/                     # Unsupervised Clustering & Spatial Analysis
│   │   ├── __init__.py
│   │   ├── kmeans.py                   # K-Means & Elbow Curve generator
│   │   ├── dbscan.py                   # DBSCAN density clustering & noise breakdown
│   │   ├── knee_for_dbscan.py          # Automated Knee eps detection from k-distance graph
│   │   ├── hierarchical.py             # Agglomerative clustering & tree dendrograms
│   │   └── geographic.py               # Spatial accident density analysis
│   │
│   ├── evaluation/                     # Unfabricated performance metrics
│   │   ├── __init__.py
│   │   ├── classification.py           # Accuracy, Precision, Recall, F1, ROC-AUC
│   │   ├── regression.py               # R², Adjusted R², RMSE, MAE, MSE
│   │   └── clustering.py               # Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia
│   │
│   ├── services/                       # Application Services
│   │   ├── __init__.py
│   │   ├── model_manager.py            # Thread-safe in-memory model artifact manager
│   │   └── prediction_service.py       # Centralized prediction handler
│   │
│   ├── templates/                      # Jinja2 HTML Templates
│   │   ├── index.html                  # Master Dashboard & Data Loading UI
│   │   ├── eda.html                    # 14-chart visual analytics gallery
│   │   ├── preprocessing.html          # Preprocessing transformations & metrics
│   │   ├── linear_regression.html      # OLS & regularized regression studio
│   │   ├── logistic_regression.html    # Probability classifier & confusion matrix
│   │   ├── decision_trees.html         # Tree depth simulator & 8-model selector
│   │   └── clustering.html             # K-Means, DBSCAN, Hierarchical & Geo Studio
│   │
│   └── static/                         # Frontend Static Assets
│       ├── css/style.css               # Design tokens, glassmorphism, dark/light themes
│       ├── images/favicon.svg          # Application favicon
│       └── charts/                     # Generated PNG charts & results cache
│
├── data/                               # Dataset repository (Isolated from source code)
│   ├── raw/
│   │   └── US_Accidents_Sample_100k.csv # 100k representative sample
│   ├── processed/
│   │   └── roadsafety_preprocessed.csv  # Cleaned, imputed, and scaled matrix
│   └── README.md                       # Large dataset setup guide
│
├── artifacts/                          # Serialized model objects (.pkl binaries)
│   ├── models/                         # Serialized classifiers and regressors
│   ├── scalers/                        # Fitted StandardScaler object
│   ├── encoders/                       # Fitted SimpleImputer object
│   └── README.md                       # Artifact documentation
│
├── scripts/                            # Standalone CLI Runners
│   ├── preprocess_data.py              # Executes data preprocessing pipeline
│   ├── train_models.py                 # Trains and benchmarks all ML models
│   └── run_clustering.py               # Executes clustering algorithms via CLI
│
├── notebooks/                          # Jupyter Notebooks & Prototyping
│   ├── eda/                            # Exploratory notebooks
│   ├── experiments/                    # Prototyping scratchpads
│   └── README.md
│
├── tests/                              # Automated Pytest Test Suite
│   ├── __init__.py
│   ├── test_app.py                     # Flask route, template & API integration tests
│   ├── test_models.py                  # Supervised ML model tests
│   ├── test_preprocessing.py           # Preprocessing & encoding tests
│   ├── test_clustering.py              # Clustering & automated knee tests
│   └── test_data.py                    # Data loader & validator tests
│
├── docs/                               # Comprehensive Documentation
│   ├── architecture.md                 # Layered architecture & caching design
│   ├── ml_methods.md                   # Mathematical formulations & theory
│   ├── api.md                          # Complete REST API reference
│   └── development.md                  # Development & contribution guide
│
└── frontend/                           # Optional React 18 + Vite SPA interface
    ├── package.json
    ├── vite.config.js
    └── src/
```

---

## 🌐 Web Studio Navigation & Endpoints

| Route | View Description | Key Functionalities |
| :--- | :--- | :--- |
| **`/`** | Master Dashboard | Overview metrics, workflow guide, feature directory, and quick links. |
| **`/data-loading`** | Data Ingestion | Dataset schema inspector, row counter, memory profiling, and column metadata. |
| **`/eda`** | Exploratory Data Analysis | 14 Interactive charts, environmental correlation heatmaps, severity distributions. |
| **`/preprocessing`** | Preprocessing Lab | IQR outlier boundaries, scaling charts, and imputation comparison tables. |
| **`/linear-regression`** | Linear Regression Studio | Continuous risk prediction, regularizer comparisons ($L_1$, $L_2$, ElasticNet), scenario presets. |
| **`/logistic-regression`** | Logistic Classifier | Probability output, odds ratio weights, and dynamic Confusion Matrix cards. |
| **`/decision-trees`** | Decision Tree Studio | Non-linear risk classification, tree depth tuning, and feature importance rankings across 8 models. |
| **`/clustering`** | Clustering Studio | K-Means (Elbow curve), DBSCAN (auto-knee $\epsilon$), Hierarchical dendrograms, and Geographic density. |

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/data-summary` | `GET` | Returns dataset profiling, completeness %, and column metadata. |
| `/api/eda` | `GET` | Returns EDA summary metrics and chart catalog. |
| `/api/preprocessing` | `GET` | Returns preprocessing metrics, outlier bounds, and imputation benchmark tables. |
| `/api/models/comparison` | `GET` | Returns benchmark leaderboard for all regression and classification models. |
| `/api/predict/linear` | `POST` | Inferences continuous impact distance for input telemetry parameters. |
| `/api/predict/logistic` | `POST` | Inferences binary accident severity tier and probability. |
| `/api/clustering/kmeans` | `POST` | Executes K-Means clustering for given $k$ and returns inertia & silhouette scores. |
| `/api/clustering/dbscan` | `POST` | Executes DBSCAN clustering with manual or auto-knee $\epsilon$. |
| `/api/clustering/knee` | `POST` | Calculates sorted $k$-distance curve and detects optimal Knee $\epsilon$. |
| `/api/clustering/hierarchical`| `POST` | Runs Agglomerative clustering with specified linkage criterion. |
| `/api/charts/<filename>` | `GET` | Serves high-resolution visual PNG figures. |

---

## 🧪 Automated Testing

Execute the automated test suite with pytest:

```bash
python -m pytest -v
```

All 32 test cases across routes, preprocessing, models, clustering, and data validation are verified and pass out of the box.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. **Fork** the project repository.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add awesome feature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
  <sub>Engineered by <a href="https://github.com/jaideep072">Jaideep</a>. Built for data-driven road safety intelligence.</sub>
</div>
