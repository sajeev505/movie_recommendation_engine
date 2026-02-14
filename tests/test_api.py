"""Tests for the Flask API endpoints."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client():
    """Create test client."""
    os.environ["DEMO_MODE"] = "true"
    from src.backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "healthy"


class TestMetricsEndpoint:
    def test_metrics_returns_json(self, client):
        res = client.get("/api/metrics")
        # May be 200 or 503 depending on whether model is loaded
        assert res.status_code in (200, 503)


class TestRecommendationsEndpoint:
    def test_recommendations_for_valid_user(self, client):
        res = client.get("/api/recommendations/1?n=5")
        if res.status_code == 503:
            pytest.skip("Model not loaded")
        data = res.get_json()
        assert res.status_code == 200
        assert "recommendations" in data
        assert data["count"] <= 5
        assert "latency_ms" in data

    def test_recommendations_fields(self, client):
        res = client.get("/api/recommendations/1?n=1")
        if res.status_code == 503:
            pytest.skip("Model not loaded")
        data = res.get_json()
        rec = data["recommendations"][0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert "rationale" in rec

    def test_recommendations_max_n(self, client):
        """N should be capped at 50."""
        res = client.get("/api/recommendations/1?n=100")
        if res.status_code == 503:
            pytest.skip("Model not loaded")
        data = res.get_json()
        assert data["count"] <= 50


class TestSearchEndpoint:
    def test_search_requires_query(self, client):
        res = client.get("/api/movies/search")
        assert res.status_code == 400

    def test_search_min_length(self, client):
        res = client.get("/api/movies/search?q=a")
        assert res.status_code == 400

    def test_search_returns_results(self, client):
        res = client.get("/api/movies/search?q=matrix")
        if res.status_code == 503:
            pytest.skip("Model not loaded")
        data = res.get_json()
        assert res.status_code == 200
        assert "movies" in data


class TestAuthEndpoints:
    def test_demo_login(self, client):
        res = client.get("/auth/demo")
        assert res.status_code == 200
        data = res.get_json()
        assert data["user"]["is_demo"] is True

    def test_auth_me_unauthenticated(self, client):
        res = client.get("/auth/me")
        assert res.status_code == 200
        data = res.get_json()
        assert data["authenticated"] is False

    def test_logout(self, client):
        # Login first
        client.get("/auth/demo")
        # Then logout
        res = client.get("/auth/logout")
        assert res.status_code == 200


class TestUserRatingsEndpoint:
    def test_user_ratings(self, client):
        res = client.get("/api/user/1/ratings")
        if res.status_code == 503:
            pytest.skip("Model not loaded")
        data = res.get_json()
        assert res.status_code == 200
        assert "ratings" in data
