"""
RoadSafety_ML Application Runner.
Entry point to launch the Flask Web Studio and REST API server.
Usage: python run.py
"""

import os
from app import create_app
from app.config import Config

app = create_app(Config)

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")

    print("==================================================")
    print("  🚦 RoadSafety AI Web Studio & REST API Server")
    print(f"  Access URL: http://{host}:{port}")
    print("==================================================")

    app.run(host=host, port=port, debug=debug)
