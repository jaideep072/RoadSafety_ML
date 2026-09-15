import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from load_data import get_data_summary
from roadsafety_eda import run_eda

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/download/preprocessed-csv")
def download_preprocessed_csv():
    """Serves the clean preprocessed dataset as a downloadable CSV file."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "roadsafety_preprocessed.csv")
    if not os.path.exists(csv_path):
        from preprocessing_pipeline import run_preprocessing_pipeline
        run_preprocessing_pipeline()
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="roadsafety_preprocessed.csv"
    )


@app.route("/data-loading")
def data_loading():
    """Loads either original or preprocessed dataset and renders summary into the page."""
    error = None
    summary = None
    dataset_type = request.args.get("dataset", "original")
    try:
        summary = get_data_summary(dataset_type=dataset_type)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        dataset_type=dataset_type,
        error=error,
    )


@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        eda_output = run_eda()
    except FileNotFoundError as e:
        try:
            eda_output = run_eda(force_run=True)
        except Exception as retry_error:
            error = f"Unexpected error: {retry_error}"
    except Exception as e:
        try:
            eda_output = run_eda(force_run=True)
        except Exception as retry_error:
            error = f"Unexpected error: {retry_error}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing():
    """Runs the preprocessing pipeline step-by-step and displays results."""
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


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    """Runs Linear Regression predicting Distance(mi) with dropdown for Without Regularization, Ridge (L2), and Lasso (L1)."""
    from linear_regression_model import train_and_evaluate_linear
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
        error = f"Error evaluating Linear Regression: {e}"
        
    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        error=error,
    )


@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():
    """Runs Logistic Regression predicting Severity (Severe vs Minor) with dropdown for Without Regularization, Ridge (L2), and Lasso (L1)."""
    from logistic_regression_model import train_and_evaluate_logistic
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
        error = f"Error evaluating Logistic Regression: {e}"
        
    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=results,
        error=error,
    )


@app.route("/decision-trees", methods=["GET", "POST"])
def decision_trees():
    """Runs Decision Trees and Ensembles with dropdown for all 7 algorithms in progressive order."""
    from decision_trees_model import train_and_evaluate_tree
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
        error = f"Error evaluating Decision Tree / Ensemble: {e}"
        
    return render_template(
        "decision_trees.html",
        active="decision-trees",
        results=results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5004)
