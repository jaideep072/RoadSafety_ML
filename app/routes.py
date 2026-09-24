"""
RoadSafety_ML Application Routes and REST API Blueprint.
Handles server-rendered HTML pages, forms, and JSON API endpoints.
"""

import os
import json
import logging
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, send_file, current_app

from app.config import (
    PROCESSED_DATA_PATH,
    STATIC_DIR,
    CHARTS_DIR,
    MODELS_CACHE_PATH
)
from app.data.loader import get_data_summary, load_dataset
from app.eda import run_eda
from app.preprocessing.pipeline import run_preprocessing_pipeline
from app.models.linear_regression import train_and_evaluate_linear
from app.models.logistic_regression import train_and_evaluate_logistic
from app.models.decision_tree import train_and_evaluate_tree
from app.models.pipeline import run_models_pipeline
from app.clustering.kmeans import run_kmeans_clustering
from app.clustering.dbscan import run_dbscan_clustering
from app.clustering.knee_for_dbscan import find_knee_for_dbscan
from app.clustering.hierarchical import run_hierarchical_clustering
from app.clustering.geographic import run_geographic_analysis
from app.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


# ==============================================================================
# HTML Template Views
# ==============================================================================

@main_bp.route("/")
def index():
    """Master Dashboard landing page."""
    return render_template("index.html", active="none")


@main_bp.route("/download/preprocessed-csv")
def download_preprocessed_csv():
    """Serves the clean preprocessed dataset as a downloadable CSV file."""
    if not PROCESSED_DATA_PATH.exists():
        run_preprocessing_pipeline()
    return send_file(
        str(PROCESSED_DATA_PATH),
        mimetype="text/csv",
        as_attachment=True,
        download_name="roadsafety_preprocessed.csv"
    )


@main_bp.route("/data-loading")
def data_loading():
    """Loads original or preprocessed dataset summary and renders the inspection page."""
    error = None
    summary = None
    dataset_type = request.args.get("dataset", "original")
    try:
        summary = get_data_summary(dataset_type=dataset_type)
    except Exception as e:
        logger.error(f"Error in data_loading route: {e}")
        error = f"Error loading dataset: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        dataset_type=dataset_type,
        error=error,
    )


@main_bp.route("/eda")
def eda():
    """Renders 14+ statistical charts and correlation heatmaps."""
    error = None
    eda_output = None
    try:
        eda_output = run_eda()
    except Exception as e:
        logger.warning(f"Initial EDA retrieval failed: {e}. Retrying with force_run=True...")
        try:
            eda_output = run_eda(force_run=True)
        except Exception as retry_error:
            logger.error(f"EDA failed: {retry_error}")
            error = f"Error generating EDA: {retry_error}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )


@main_bp.route("/preprocessing")
def preprocessing():
    """Renders data preprocessing transformations and imputation benchmarks."""
    error = None
    preprocess_output = None
    try:
        preprocess_output = run_preprocessing_pipeline()
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        error = f"Error generating preprocessing results: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocess_output,
        error=error,
    )


@main_bp.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    """Linear Regression Studio predicting accident distance with OLS, Ridge, Lasso, and ElasticNet."""
    error = None
    results = None
    reg_type = request.values.get("reg_type", "none")

    student_input = None
    if request.method == "POST":
        student_input = {
            "Temperature(F)": request.form.get("Temperature(F)", 70.0),
            "Humidity(%)": request.form.get("Humidity(%)", 60.0),
            "Pressure(in)": request.form.get("Pressure(in)", 29.92),
            "Visibility(mi)": request.form.get("Visibility(mi)", 10.0),
            "Wind_Speed(mph)": request.form.get("Wind_Speed(mph)", 8.0),
            "Crossing": request.form.get("Crossing", 0),
            "Junction": request.form.get("Junction", 0),
            "Traffic_Signal": request.form.get("Traffic_Signal", 0)
        }

    try:
        results = train_and_evaluate_linear(reg_type=reg_type, student_input=student_input)
    except Exception as e:
        logger.error(f"Linear regression evaluation error: {e}")
        error = f"Error evaluating Linear Regression: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        error=error,
    )


@main_bp.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():
    """Logistic Regression Studio for binary accident severity tier classification."""
    error = None
    results = None
    reg_type = request.values.get("reg_type", "none")

    student_input = None
    if request.method == "POST":
        student_input = {
            "Temperature(F)": request.form.get("Temperature(F)", 70.0),
            "Humidity(%)": request.form.get("Humidity(%)", 60.0),
            "Pressure(in)": request.form.get("Pressure(in)", 29.92),
            "Visibility(mi)": request.form.get("Visibility(mi)", 10.0),
            "Wind_Speed(mph)": request.form.get("Wind_Speed(mph)", 8.0),
            "Crossing": request.form.get("Crossing", 0),
            "Junction": request.form.get("Junction", 0),
            "Traffic_Signal": request.form.get("Traffic_Signal", 0),
            "Sunrise_Sunset_Day": request.form.get("Sunrise_Sunset_Day", 1)
        }

    try:
        results = train_and_evaluate_logistic(reg_type=reg_type, student_input=student_input)
    except Exception as e:
        logger.error(f"Logistic regression evaluation error: {e}")
        error = f"Error evaluating Logistic Regression: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=results,
        error=error,
    )


@main_bp.route("/decision-trees", methods=["GET", "POST"])
def decision_trees():
    """Decision Trees & Ensembles Studio supporting 7 tree algorithms."""
    error = None
    results = None
    algo = request.values.get("algo", "dt")

    student_input = None
    if request.method == "POST":
        student_input = {
            "Temperature(F)": request.form.get("Temperature(F)", 70.0),
            "Humidity(%)": request.form.get("Humidity(%)", 60.0),
            "Pressure(in)": request.form.get("Pressure(in)", 29.92),
            "Visibility(mi)": request.form.get("Visibility(mi)", 10.0),
            "Wind_Speed(mph)": request.form.get("Wind_Speed(mph)", 8.0),
            "Crossing": request.form.get("Crossing", 0),
            "Junction": request.form.get("Junction", 0),
            "Traffic_Signal": request.form.get("Traffic_Signal", 0),
            "Sunrise_Sunset_Day": request.form.get("Sunrise_Sunset_Day", 1)
        }

    try:
        results = train_and_evaluate_tree(algo_key=algo, student_input=student_input)
    except Exception as e:
        logger.error(f"Decision tree evaluation error: {e}")
        error = f"Error evaluating Tree Model: {e}"

    return render_template(
        "decision_trees.html",
        active="decision-trees",
        results=results,
        error=error,
    )


@main_bp.route("/clustering", methods=["GET", "POST"])
def clustering():
    """Unsupervised Clustering Studio: K-Means, DBSCAN (Auto-Knee), and Hierarchical Clustering."""
    error = None
    results = None
    active_tab = request.values.get("algo", "kmeans")

    try:
        df = load_dataset(dataset_type="preprocessed")

        if active_tab == "dbscan":
            min_samples = int(request.values.get("min_samples", 5))
            eps_val = request.values.get("eps", "")
            auto_eps = (eps_val == "" or request.values.get("auto_eps") == "true")
            parsed_eps = float(eps_val) if (eps_val != "" and not auto_eps) else None

            results = run_dbscan_clustering(
                df=df,
                eps=parsed_eps,
                min_samples=min_samples,
                auto_eps=auto_eps
            )
        elif active_tab == "hierarchical":
            n_clusters = int(request.values.get("n_clusters", 3))
            linkage_type = request.values.get("linkage", "ward")
            results = run_hierarchical_clustering(
                df=df,
                n_clusters=n_clusters,
                linkage_type=linkage_type
            )
        elif active_tab == "geographic":
            results = run_geographic_analysis(df=df)
        else:
            # Default K-Means
            active_tab = "kmeans"
            n_clusters = int(request.values.get("n_clusters", 4))
            results = run_kmeans_clustering(
                df=df,
                n_clusters=n_clusters,
                compute_elbow=True
            )
    except Exception as e:
        logger.error(f"Clustering route error: {e}")
        error = f"Clustering computation error: {e}"

    return render_template(
        "clustering.html",
        active="clustering",
        active_tab=active_tab,
        results=results,
        error=error,
    )


# ==============================================================================
# REST API Endpoints (For JSON Consumers & React Frontend)
# ==============================================================================

@main_bp.route("/api/data-summary")
def api_data_summary():
    """Returns dataset summary statistics and column metadata."""
    dataset_type = request.args.get("dataset", "original")
    try:
        summary = get_data_summary(dataset_type=dataset_type)
        return jsonify({"status": "success", "data": summary})
    except Exception as e:
        logger.error(f"API Data summary error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/api/eda")
def api_eda():
    """Returns EDA summary statistics and chart catalog."""
    try:
        eda_output = run_eda()
        return jsonify({"status": "success", "data": eda_output})
    except Exception as e:
        logger.error(f"API EDA error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/api/preprocessing")
def api_preprocessing():
    """Returns preprocessing diagnostics and imputation comparisons."""
    try:
        preprocess_output = run_preprocessing_pipeline()
        return jsonify({"status": "success", "data": preprocess_output})
    except Exception as e:
        logger.error(f"API Preprocessing error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/api/models/comparison")
def api_models_comparison():
    """Returns comparative model benchmark leaderboard for all algorithms."""
    try:
        cache = run_models_pipeline()
        return jsonify({"status": "success", "data": cache})
    except Exception as e:
        logger.error(f"API Models comparison error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/api/predict/linear", methods=["POST"])
def api_predict_linear():
    """Infers continuous delay/impact distance."""
    data = request.get_json() or {}
    model_name = data.get("model", "linear_reg")
    features = data.get("features", {})
    try:
        res = prediction_service.predict_regression(model_name, features)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API Linear predict error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/predict/logistic", methods=["POST"])
def api_predict_logistic():
    """Infers binary accident severity classification."""
    data = request.get_json() or {}
    model_name = data.get("model", "logistic_reg")
    features = data.get("features", {})
    try:
        res = prediction_service.predict_classification(model_name, features)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API Logistic predict error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/clustering/kmeans", methods=["POST"])
def api_clustering_kmeans():
    """Runs K-Means clustering via API."""
    data = request.get_json() or {}
    n_clusters = int(data.get("n_clusters", 4))
    features = data.get("features", None)
    try:
        df = load_dataset(dataset_type="preprocessed")
        res = run_kmeans_clustering(df, feature_cols=features, n_clusters=n_clusters)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API K-Means error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/clustering/dbscan", methods=["POST"])
def api_clustering_dbscan():
    """Runs DBSCAN clustering via API."""
    data = request.get_json() or {}
    eps = float(data.get("eps", 0.5)) if "eps" in data and data["eps"] is not None else None
    min_samples = int(data.get("min_samples", 5))
    auto_eps = data.get("auto_eps", eps is None)
    features = data.get("features", None)
    try:
        df = load_dataset(dataset_type="preprocessed")
        res = run_dbscan_clustering(df, feature_cols=features, eps=eps, min_samples=min_samples, auto_eps=auto_eps)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API DBSCAN error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/clustering/knee", methods=["POST"])
def api_clustering_knee():
    """Runs Knee detection for DBSCAN via API."""
    data = request.get_json() or {}
    min_samples = int(data.get("min_samples", 5))
    try:
        df = load_dataset(dataset_type="preprocessed")
        features = data.get("features", ["Start_Lng", "Start_Lat", "Temperature(F)", "Humidity(%)", "Pressure(in)"])
        active_features = [f for f in features if f in df.columns]
        res = find_knee_for_dbscan(df[active_features].dropna().values, min_samples=min_samples)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API Knee error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/clustering/hierarchical", methods=["POST"])
def api_clustering_hierarchical():
    """Runs Hierarchical clustering via API."""
    data = request.get_json() or {}
    n_clusters = int(data.get("n_clusters", 3))
    linkage = data.get("linkage", "ward")
    features = data.get("features", None)
    try:
        df = load_dataset(dataset_type="preprocessed")
        res = run_hierarchical_clustering(df, feature_cols=features, n_clusters=n_clusters, linkage_type=linkage)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"API Hierarchical error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400


@main_bp.route("/api/charts/<path:filename>")
def api_charts(filename):
    """Serves generated PNG charts from static/charts."""
    chart_path = CHARTS_DIR / filename
    if not chart_path.exists():
        chart_path = STATIC_DIR / filename
    if not chart_path.exists():
        return jsonify({"status": "error", "message": "Chart not found"}), 404
    return send_file(str(chart_path), mimetype="image/png")
