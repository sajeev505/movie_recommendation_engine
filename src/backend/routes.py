"""API route definitions."""

import time
from flask import Blueprint, request, jsonify, session
from .auth import login_required

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Will be set by app.py on startup
recommender = None


def init_routes(rec):
    """Initialize routes with the recommender instance."""
    global recommender
    recommender = rec


@api_bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": recommender is not None,
    })


@api_bp.route("/metrics")
def metrics():
    """Return model performance metrics."""
    if not recommender:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(recommender.metrics)


@api_bp.route("/recommendations/<int:user_id>")
def get_recommendations(user_id):
    """
    Get top-N movie recommendations for a user.

    Query params:
        n (int): Number of recommendations (default 10, max 50)

    Returns:
        JSON array of recommended movies with scores and rationales.
    """
    if not recommender:
        return jsonify({"error": "Model not loaded"}), 503

    n = min(int(request.args.get("n", 10)), 50)

    results, latency_ms = recommender.get_recommendations(user_id, n=n)

    return jsonify({
        "user_id": user_id,
        "count": len(results),
        "latency_ms": round(latency_ms, 1),
        "recommendations": results,
    })


@api_bp.route("/movies/search")
def search_movies():
    """
    Search movies by title.

    Query params:
        q (str): Search query
        limit (int): Max results (default 20)
    """
    if not recommender:
        return jsonify({"error": "Model not loaded"}), 503

    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify({"error": "Query must be at least 2 characters"}), 400

    limit = min(int(request.args.get("limit", 20)), 50)
    results = recommender.search_movies(query, limit=limit)

    return jsonify({
        "query": query,
        "count": len(results),
        "movies": results,
    })


@api_bp.route("/user/<int:user_id>/ratings")
def user_ratings(user_id):
    """Get a user's existing ratings."""
    if not recommender:
        return jsonify({"error": "Model not loaded"}), 503

    ratings = recommender.get_user_ratings(user_id)
    return jsonify({
        "user_id": user_id,
        "count": len(ratings),
        "ratings": ratings,
    })


@api_bp.route("/users")
def list_users():
    """Get sample user IDs for demo purposes."""
    if not recommender:
        return jsonify({"error": "Model not loaded"}), 503

    all_users = recommender.get_all_user_ids()
    # Return first 20 user IDs as sample
    sample = sorted(all_users[:20])
    return jsonify({
        "total_users": len(all_users),
        "sample_user_ids": sample,
    })
