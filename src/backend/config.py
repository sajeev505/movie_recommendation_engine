"""Flask application configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

    # GitHub OAuth
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_API_URL = "https://api.github.com/user"

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "svd_model.pkl"
    ))
    METRICS_PATH = os.getenv("METRICS_PATH", os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "metrics.json"
    ))
    MOVIES_PATH = os.getenv("MOVIES_PATH", os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "movies_metadata.csv"
    ))

    # API
    API_URL = os.getenv("API_URL", "http://localhost:5000")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Demo user — maps to a real MovieLens user with diverse ratings
    DEMO_USER_ID = 1
    DEMO_USER_NAME = "Demo Reviewer"
