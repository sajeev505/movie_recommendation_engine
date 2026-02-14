"""Flask application factory — Movie Recommendation Engine API."""

import os
import sys

from flask import Flask
from flask_cors import CORS

from .config import Config
from .auth import auth_bp
from .routes import api_bp, init_routes
from .recommender import Recommender


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = Config.SECRET_KEY

    # CORS — allow frontend origin
    CORS(app, supports_credentials=True, origins=[
        Config.FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
    ])

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # Load model and initialize recommender
    try:
        rec = Recommender(
            model_path=Config.MODEL_PATH,
            movies_path=Config.MOVIES_PATH,
            metrics_path=Config.METRICS_PATH,
        )
        init_routes(rec)
        print("✓ Recommender engine loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠ WARNING: {e}")
        print("  API will start but /api/recommendations will return 503.")
        print("  Run: python scripts/download_data.py && "
              "python scripts/preprocess.py && python scripts/train_model.py")

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Endpoint not found"}, 404

    @app.errorhandler(500)
    def internal_error(e):
        return {"error": "Internal server error"}, 500

    return app


# For `flask run` or `gunicorn src.backend.app:app`
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
