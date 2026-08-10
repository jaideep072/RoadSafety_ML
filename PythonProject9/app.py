import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from load_data import get_data_summary
from roadsafety_eda import run_eda

app = Flask(__name__)
# Enable CORS for all routes (to allow React to fetch data during development)
CORS(app)

@app.route("/api/data-summary")
def data_loading():
    """Returns the dataset summary as JSON."""
    try:
        summary = get_data_summary()
        return jsonify({"success": True, "data": summary})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500

@app.route("/api/eda")
def eda():
    """Runs exploratory data analysis and returns results as JSON."""
    try:
        eda_output = run_eda()
        return jsonify({"success": True, "data": eda_output})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500

@app.route("/api/charts/<path:filename>")
def serve_charts(filename):
    """Serves the generated charts for the frontend to display."""
    charts_dir = os.path.join(os.path.dirname(__file__), "static", "charts")
    return send_from_directory(charts_dir, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
