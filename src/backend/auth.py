"""GitHub OAuth and demo-mode authentication."""

import os
import functools
import requests
from flask import Blueprint, request, session, redirect, jsonify, url_for
from .config import Config

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def login_required(f):
    """Decorator to require authentication (or demo mode)."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if Config.DEMO_MODE:
            session.setdefault("user", {
                "id": Config.DEMO_USER_ID,
                "login": Config.DEMO_USER_NAME,
                "avatar_url": None,
                "is_demo": True,
            })
        if "user" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return wrapper


@auth_bp.route("/github")
def github_login():
    """Redirect to GitHub OAuth authorization."""
    if Config.DEMO_MODE:
        return redirect(url_for("auth.demo_login"))

    if not Config.GITHUB_CLIENT_ID:
        return jsonify({"error": "GitHub OAuth not configured. Use demo mode."}), 500

    params = {
        "client_id": Config.GITHUB_CLIENT_ID,
        "scope": "read:user",
        "redirect_uri": f"{Config.API_URL}/auth/github/callback",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return redirect(f"{Config.GITHUB_AUTH_URL}?{query}")


@auth_bp.route("/github/callback")
def github_callback():
    """Handle GitHub OAuth callback — exchange code for token."""
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "No authorization code provided"}), 400

    # Exchange code for access token
    token_resp = requests.post(
        Config.GITHUB_TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": Config.GITHUB_CLIENT_ID,
            "client_secret": Config.GITHUB_CLIENT_SECRET,
            "code": code,
        },
        timeout=10,
    )
    token_data = token_resp.json()
    access_token = token_data.get("access_token")

    if not access_token:
        return jsonify({"error": "Failed to obtain access token"}), 400

    # Fetch user profile
    user_resp = requests.get(
        Config.GITHUB_API_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    user_data = user_resp.json()

    # Store user in session — map GitHub user to a MovieLens user for recs
    session["user"] = {
        "id": Config.DEMO_USER_ID,  # all OAuth users get same demo recs
        "login": user_data.get("login", "github-user"),
        "avatar_url": user_data.get("avatar_url"),
        "github_id": user_data.get("id"),
        "is_demo": False,
    }

    return redirect(Config.FRONTEND_URL)


@auth_bp.route("/demo")
def demo_login():
    """Log in as a demo user (no OAuth required)."""
    session["user"] = {
        "id": Config.DEMO_USER_ID,
        "login": Config.DEMO_USER_NAME,
        "avatar_url": None,
        "is_demo": True,
    }
    return jsonify({"message": "Logged in as demo user", "user": session["user"]})


@auth_bp.route("/me")
def current_user():
    """Get current authenticated user info."""
    user = session.get("user")
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": user})


@auth_bp.route("/logout")
def logout():
    """Clear session."""
    session.pop("user", None)
    return jsonify({"message": "Logged out"})
