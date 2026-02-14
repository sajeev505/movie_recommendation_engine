"""Recommendation engine — loads trained SVD model and generates top-N predictions."""

import os
import time
import pickle
import json

import pandas as pd
import numpy as np


class Recommender:
    """SVD-based collaborative filtering recommender (scikit-learn backend)."""

    def __init__(self, model_path, movies_path, metrics_path):
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.user_means = None
        self.user_ids = []
        self.movie_ids = []
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.user_rated = {}
        self.movies_df = None
        self.metrics = {}
        self._movie_lookup = {}
        self._genre_lookup = {}

        self._load_model(model_path)
        self._load_movies(movies_path)
        self._load_metrics(metrics_path)

    def _load_model(self, model_path):
        """Load trained SVD model artefacts from pickle."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run scripts/train_model.py first."
            )
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.global_mean = data["global_mean"]
        self.user_means = data["user_means"]
        self.user_ids = data["user_ids"]
        self.movie_ids = data["movie_ids"]
        self.user_to_idx = data["user_to_idx"]
        self.movie_to_idx = data["movie_to_idx"]
        self.user_rated = data["user_rated"]

        print(f"Model loaded: {len(self.user_ids)} users, "
              f"{len(self.movie_ids)} items")

    def _load_movies(self, movies_path):
        """Load movie metadata for enriching recommendations."""
        if not os.path.exists(movies_path):
            raise FileNotFoundError(
                f"Movies metadata not found at {movies_path}. "
                "Run scripts/preprocess.py first."
            )
        self.movies_df = pd.read_csv(movies_path)
        for _, row in self.movies_df.iterrows():
            mid = int(row["movieId"])
            self._movie_lookup[mid] = {
                "movie_id": mid,
                "title": row["clean_title"],
                "original_title": row["title"],
                "year": str(row["year"]) if pd.notna(row["year"]) else None,
                "genres": row["genres"].split("|") if pd.notna(row["genres"]) else [],
                "tmdb_id": int(row["tmdbId"]) if pd.notna(row["tmdbId"]) and row["tmdbId"] != 0 else None,
            }
            self._genre_lookup[mid] = row["genres"] if pd.notna(row["genres"]) else ""
        print(f"Movie metadata loaded: {len(self._movie_lookup)} movies")

    def _load_metrics(self, metrics_path):
        """Load training metrics for the API to serve."""
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            print(f"Metrics loaded: MAE reduction = {self.metrics.get('mae_reduction_pct', '?')}%")
        else:
            self.metrics = {}

    def _predict_score(self, user_idx, movie_idx):
        """Predict rating for a single (user, movie) pair using latent factors."""
        score = self.user_means[user_idx] + \
                np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        return float(np.clip(score, 0.5, 5.0))

    def get_recommendations(self, user_id, n=10):
        """
        Generate top-N movie recommendations for a given user.

        Returns list of dicts with movie_id, title, year, genres,
        predicted score, and a brief rationale.
        """
        start_time = time.time()

        if user_id not in self.user_to_idx:
            # Unknown user — return popular movies as fallback
            rated_items = set()
            user_idx = None
        else:
            user_idx = self.user_to_idx[user_id]
            rated_items = self.user_rated.get(user_id, set())

        # Predict ratings for all unrated movies
        predictions = []
        for movie_id in self._movie_lookup:
            if movie_id in rated_items:
                continue
            if movie_id not in self.movie_to_idx:
                continue

            if user_idx is not None:
                movie_idx = self.movie_to_idx[movie_id]
                score = self._predict_score(user_idx, movie_idx)
            else:
                score = self.global_mean

            predictions.append((movie_id, score))

        # Sort by predicted rating (descending) and take top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]

        # Enrich with metadata
        results = []
        for movie_id, score in top_n:
            movie = self._movie_lookup.get(movie_id, {})
            genres = movie.get("genres", [])
            rationale = self._generate_rationale(user_id, movie_id, genres, score)

            results.append({
                "movie_id": movie_id,
                "title": movie.get("title", f"Movie {movie_id}"),
                "year": movie.get("year"),
                "genres": genres,
                "score": round(score, 2),
                "rationale": rationale,
                "poster_url": self._get_poster_url(movie.get("tmdb_id")),
            })

        elapsed_ms = (time.time() - start_time) * 1000
        return results, elapsed_ms

    def _generate_rationale(self, user_id, movie_id, genres, score):
        """Generate a brief, human-readable recommendation rationale."""
        if score >= 4.5:
            strength = "an excellent"
        elif score >= 4.0:
            strength = "a strong"
        elif score >= 3.5:
            strength = "a good"
        else:
            strength = "a reasonable"

        genre_text = ", ".join(genres[:2]) if genres else "this genre"
        return (
            f"Predicted as {strength} match (score: {score:.1f}/5). "
            f"Based on your rating patterns for {genre_text} films."
        )

    def _get_poster_url(self, tmdb_id):
        """Return a TMDB poster URL placeholder (requires TMDB API key for actual images)."""
        if tmdb_id:
            return f"https://image.tmdb.org/t/p/w300/{tmdb_id}"
        return None

    def search_movies(self, query, limit=20):
        """Search movies by title (case-insensitive substring match)."""
        query_lower = query.lower()
        results = []
        for movie_id, movie in self._movie_lookup.items():
            if query_lower in movie["title"].lower():
                results.append(movie)
                if len(results) >= limit:
                    break
        return results

    def get_user_ratings(self, user_id, limit=20):
        """Get a user's existing ratings with movie metadata."""
        if user_id not in self.user_to_idx:
            return []

        rated_movies = self.user_rated.get(user_id, set())
        if not rated_movies:
            return []

        # We don't store the actual ratings in the pickle (only the model),
        # so we reload from the processed CSV if needed.
        ratings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "processed", "all_ratings.csv"
        )
        if not os.path.exists(ratings_path):
            return []

        df = pd.read_csv(ratings_path)
        user_df = df[df["userId"] == user_id].sort_values("rating", ascending=False)

        ratings = []
        for _, row in user_df.head(limit).iterrows():
            mid = int(row["movieId"])
            movie = self._movie_lookup.get(mid, {})
            ratings.append({
                "movie_id": mid,
                "title": movie.get("title", f"Movie {mid}"),
                "year": movie.get("year"),
                "genres": movie.get("genres", []),
                "rating": row["rating"],
            })

        return ratings

    def get_all_user_ids(self):
        """Return list of all known user IDs."""
        return [int(uid) for uid in self.user_ids]

    def is_valid_user(self, user_id):
        """Check if a user ID exists in the dataset."""
        return user_id in self.user_to_idx
