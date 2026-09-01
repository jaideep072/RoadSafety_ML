import os

from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS

from load_data import get_data_summary
from roadsafety_eda import run_eda


app = Flask(__name__)

# Enable CORS
CORS(app)


# =========================================================
# HOME / DATA LOADING PAGE
# =========================================================
@app.route("/")
def home():
    return data_loading()


# =========================================================
# DATA LOADING PAGE
# =========================================================
@app.route("/data-loading")
def data_loading():
    try:
        # Load dataset summary
        summary = get_data_summary()

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=None
        )

    except FileNotFoundError as e:
        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=str(e)
        )

    except Exception as e:
        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=f"Unexpected error: {e}"
        )


# =========================================================
# DATA SUMMARY API
# =========================================================
@app.route("/api/data-summary")
def data_loading_api():
    try:
        summary = get_data_summary()

        return jsonify({
            "success": True,
            "data": summary
        })

    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {e}"
        }), 500


# =========================================================
# EDA PAGE
# =========================================================
@app.route("/eda")
def eda():
    try:
        # Run EDA
        eda_output = run_eda()

        return render_template(
            "eda.html",
            results=eda_output,
            active="eda",
            error=None
        )

    except FileNotFoundError as e:
        return render_template(
            "eda.html",
            results=None,
            active="eda",
            error=str(e)
        )

    except Exception as e:
        return render_template(
            "eda.html",
            results=None,
            active="eda",
            error=f"Unexpected error: {e}"
        )


# =========================================================
# EDA API
# =========================================================
@app.route("/api/eda")
def eda_api():
    try:
        eda_output = run_eda()

        return jsonify({
            "success": True,
            "data": eda_output
        })

    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {e}"
        }), 500


# =========================================================
# SERVE EDA CHARTS
# =========================================================
@app.route("/api/charts/<filename>")
def serve_charts(filename):

    charts_dir = os.path.join(
        os.path.dirname(__file__),
        "static",
        "charts"
    )

    return send_from_directory(
        charts_dir,
        filename
    )


# =========================================================
# PREPROCESSING PAGE
# =========================================================
@app.route("/preprocessing")
def preprocessing():
    from preprocessing_pipeline import run_preprocessing_pipeline
    error = None
    preprocess_output = None
    try:
        preprocess_output = run_preprocessing_pipeline()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocess_output,
        error=error,
    )


# =========================================================
# PREPROCESSING API
# =========================================================
@app.route("/api/preprocessing")
def preprocessing_api():
    from preprocessing_pipeline import run_preprocessing_pipeline
    try:
        preprocess_output = run_preprocessing_pipeline()
        return jsonify({
            "success": True,
            "data": preprocess_output
        })
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {e}"
        }), 500


# =========================================================
# REGRESSION PAGE
# =========================================================
@app.route("/regression")
def regression_dashboard():
    from models_pipeline import train_models
    error = None
    models_output = None
    try:
        models_output = train_models()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "regression.html",
        active="regression",
        results=models_output,
        error=error,
    )


# =========================================================
# REGULARIZATION PAGE
# =========================================================
@app.route("/regularization")
def regularization_dashboard():
    from models_pipeline import train_models
    error = None
    models_output = None
    try:
        models_output = train_models()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "regularization.html",
        active="regularization",
        results=models_output,
        error=error,
    )


# =========================================================
# DECISION TREE PAGE
# =========================================================
@app.route("/decision-tree")
def decision_tree_dashboard():
    from models_pipeline import train_models
    error = None
    models_output = None
    try:
        models_output = train_models()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "decision_tree.html",
        active="decision_tree",
        results=models_output,
        error=error,
    )


# =========================================================
# MODEL PREDICTION API
# =========================================================
@app.route("/api/predict", methods=["POST"])
def predict_api():
    from flask import request
    from models_pipeline import predict_sample
    try:
        payload = request.get_json()
        features_dict = {
            "Temperature(F)": float(payload["Temperature(F)"]),
            "Humidity(%)": float(payload["Humidity(%)"]),
            "Pressure(in)": float(payload["Pressure(in)"]),
            "Visibility(mi)": float(payload["Visibility(mi)"]),
            "Wind_Speed(mph)": float(payload["Wind_Speed(mph)"])
        }
        category = payload.get("category", "regression")
        model_type = payload.get("model_type", "none")
        prediction = predict_sample(features_dict, category=category, model_type=model_type)
        return jsonify({
            "success": True,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# RUN FLASK APPLICATION
# =========================================================
if __name__ == "__main__":
    app.run(
        debug=True,
        port=5004
    )