# REST API Reference

The RoadSafety_ML backend exposes a complete set of JSON REST API endpoints for external applications, decoupled Single Page Applications (e.g., React/Vite), and automated testing pipelines.

Base URL: `http://127.0.0.1:5000`

---

## 1. Data Endpoints

### `GET /api/data-summary`
Returns memory profiling, schema analysis, missing statistics, and data previews.

- **Query Parameters**:
  - `dataset` (optional): `original` (default) or `preprocessed`
- **Response**:
```json
{
  "status": "success",
  "data": {
    "n_rows": 100000,
    "n_cols": 34,
    "formatted_rows": "100,000",
    "memory_mb": 42.15,
    "completeness_pct": 98.42,
    "columns": ["Temperature(F)", "Humidity(%)", "..."],
    "preview": [...]
  }
}
```

---

## 2. Analytics & Preprocessing Endpoints

### `GET /api/eda`
Returns EDA summary metrics and chart metadata.

### `GET /api/preprocessing`
Returns missing analysis, outlier capping thresholds, and imputation comparisons.

### `GET /api/models/comparison`
Returns the complete leaderboard comparison across all regression and classification models.

---

## 3. Inference Endpoints

### `POST /api/predict/linear`
Predicts continuous accident blockage distance (in miles).

- **Request Body**:
```json
{
  "model": "linear_reg",
  "features": {
    "Temperature(F)": 72.0,
    "Humidity(%)": 60.0,
    "Pressure(in)": 29.92,
    "Visibility(mi)": 10.0,
    "Wind_Speed(mph)": 8.0,
    "Crossing": 0,
    "Junction": 1,
    "Traffic_Signal": 0
  }
}
```
- **Response**:
```json
{
  "status": "success",
  "data": {
    "model": "linear_reg",
    "prediction": 0.428,
    "unit": "miles"
  }
}
```

### `POST /api/predict/logistic`
Predicts accident severity classification tier.

- **Request Body**:
```json
{
  "model": "logistic_reg",
  "features": {
    "Temperature(F)": 85.0,
    "Humidity(%)": 80.0,
    "Pressure(in)": 29.80,
    "Visibility(mi)": 4.0,
    "Wind_Speed(mph)": 18.0,
    "Crossing": 1,
    "Junction": 1,
    "Traffic_Signal": 1,
    "Sunrise_Sunset_Day": 0
  }
}
```
- **Response**:
```json
{
  "status": "success",
  "data": {
    "model": "logistic_reg",
    "result": {
      "class": "Severe Accident",
      "is_severe": true,
      "probability": 84.5
    }
  }
}
```

---

## 4. Clustering Endpoints

### `POST /api/clustering/kmeans`
- **Request Body**: `{"n_clusters": 4}`
- **Response**: Returns inertia, silhouette score, Davies-Bouldin, cluster distributions, and scatter plot filename.

### `POST /api/clustering/dbscan`
- **Request Body**: `{"eps": 0.45, "min_samples": 5, "auto_eps": false}`
- **Response**: Returns core points, border points, noise count, cluster counts, and scatter plot filename.

### `POST /api/clustering/knee`
- **Request Body**: `{"min_samples": 5}`
- **Response**: Returns recommended epsilon ($\epsilon$) and k-distance curve chart filename.

### `POST /api/clustering/hierarchical`
- **Request Body**: `{"n_clusters": 3, "linkage": "ward"}`
- **Response**: Returns linkage metrics, cluster distributions, and dendrogram plot filename.

---

## 5. Asset Endpoints

### `GET /api/charts/<filename>`
Serves generated matplotlib/seaborn charts with appropriate `image/png` MIME type.
