"""
RoadSafety_ML Application Factory.
Initializes Flask app, configures CORS, static assets, error handlers, and blueprints.
"""

import logging
from flask import Flask, jsonify, url_for as flask_url_for, current_app
from flask_cors import CORS
from app.config import Config, STATIC_DIR, TEMPLATES_DIR

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config) -> Flask:
    """Creates and configures an instance of the Flask application."""
    config_class.init_app()

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(TEMPLATES_DIR)
    )
    app.config.from_object(config_class)

    # Enable Cross-Origin Resource Sharing for API consumers (e.g. React client)
    CORS(app)

    # Register Routes Blueprint
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Template context processor for robust url_for resolution
    @app.context_processor
    def utility_processor():
        def smart_url_for(endpoint, **values):
            try:
                return flask_url_for(endpoint, **values)
            except Exception:
                if not endpoint.startswith("main.") and f"main.{endpoint}" in current_app.view_functions:
                    return flask_url_for(f"main.{endpoint}", **values)
                raise
        return dict(url_for=smart_url_for)

    # Global Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "status": "error",
            "code": 404,
            "message": "Resource or endpoint not found."
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "status": "error",
            "code": 500,
            "message": "An internal server error occurred."
        }), 500

    logger.info("RoadSafety_ML Flask application initialized successfully.")
    return app
