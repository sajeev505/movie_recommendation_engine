"""Tests for the recommendation engine."""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRecommender:
    """Test the Recommender class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load recommender if model exists."""
        from src.backend.recommender import Recommender
        from src.backend.config import Config

        try:
            self.rec = Recommender(
                model_path=Config.MODEL_PATH,
                movies_path=Config.MOVIES_PATH,
                metrics_path=Config.METRICS_PATH,
            )
            self.model_available = True
        except FileNotFoundError:
            self.model_available = False
            pytest.skip("Model not trained yet — run scripts/train_model.py first")

    def test_model_loaded(self):
        """Model factors should be loaded."""
        assert self.rec.user_factors is not None
        assert self.rec.item_factors is not None
        assert len(self.rec.user_to_idx) > 0

    def test_movies_loaded(self):
        """Movie metadata should be loaded."""
        assert len(self.rec._movie_lookup) > 0
        assert len(self.rec.movie_ids) > 0

    def test_get_recommendations_returns_results(self):
        """Should return N recommendations for a valid user."""
        results, latency = self.rec.get_recommendations(user_id=1, n=5)
        assert len(results) == 5
        assert latency > 0

    def test_recommendation_fields(self):
        """Each recommendation should have required fields."""
        results, _ = self.rec.get_recommendations(user_id=1, n=1)
        rec = results[0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert "rationale" in rec
        assert "genres" in rec

    def test_recommendation_scores_ordered(self):
        """Recommendations should be sorted by score descending."""
        results, _ = self.rec.get_recommendations(user_id=1, n=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recommendations_exclude_rated(self):
        """Recommendations should not include movies the user already rated."""
        results, _ = self.rec.get_recommendations(user_id=1, n=50)
        rec_ids = {r["movie_id"] for r in results}
        rated_ids = {r["movie_id"] for r in self.rec.get_user_ratings(1, limit=100)}
        assert rec_ids.isdisjoint(rated_ids)

    def test_search_movies(self):
        """Search should return matching movies."""
        results = self.rec.search_movies("Toy Story")
        assert len(results) > 0
        assert any("Toy Story" in r["title"] for r in results)

    def test_search_case_insensitive(self):
        """Search should be case-insensitive."""
        results = self.rec.search_movies("toy story")
        assert len(results) > 0

    def test_get_user_ratings(self):
        """Should return rated movies for a valid user."""
        ratings = self.rec.get_user_ratings(user_id=1)
        assert len(ratings) > 0
        assert "rating" in ratings[0]

    def test_is_valid_user(self):
        """Should correctly identify valid/invalid user IDs."""
        assert self.rec.is_valid_user(1) is True
        assert self.rec.is_valid_user(999999) is False

    def test_get_all_user_ids(self):
        """Should return list of user IDs."""
        users = self.rec.get_all_user_ids()
        assert len(users) > 0
        assert 1 in users

    def test_deterministic_recommendations(self):
        """Same user should get same recommendations."""
        results1, _ = self.rec.get_recommendations(user_id=1, n=5)
        results2, _ = self.rec.get_recommendations(user_id=1, n=5)
        ids1 = [r["movie_id"] for r in results1]
        ids2 = [r["movie_id"] for r in results2]
        assert ids1 == ids2
