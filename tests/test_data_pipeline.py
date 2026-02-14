"""Tests for data preprocessing pipeline."""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class TestDataPipeline:
    """Test that preprocessed data is valid."""

    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        if not os.path.exists(os.path.join(PROCESSED_DIR, "all_ratings.csv")):
            pytest.skip("Processed data not available — run scripts/preprocess.py first")

    def test_ratings_file_exists(self):
        assert os.path.exists(os.path.join(PROCESSED_DIR, "all_ratings.csv"))
        assert os.path.exists(os.path.join(PROCESSED_DIR, "train_ratings.csv"))
        assert os.path.exists(os.path.join(PROCESSED_DIR, "test_ratings.csv"))

    def test_movies_metadata_exists(self):
        assert os.path.exists(os.path.join(PROCESSED_DIR, "movies_metadata.csv"))

    def test_ratings_columns(self):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "all_ratings.csv"))
        assert "userId" in df.columns
        assert "movieId" in df.columns
        assert "rating" in df.columns

    def test_ratings_no_nulls(self):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "all_ratings.csv"))
        assert df[["userId", "movieId", "rating"]].isnull().sum().sum() == 0

    def test_rating_range(self):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "all_ratings.csv"))
        assert df["rating"].min() >= 0.5
        assert df["rating"].max() <= 5.0

    def test_train_test_split(self):
        train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_ratings.csv"))
        test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_ratings.csv"))
        total = pd.read_csv(os.path.join(PROCESSED_DIR, "all_ratings.csv"))
        assert len(train) + len(test) == len(total)

    def test_movies_have_titles(self):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_metadata.csv"))
        assert df["clean_title"].isnull().sum() == 0
        assert df["title"].isnull().sum() == 0

    def test_movies_have_genres(self):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_metadata.csv"))
        assert df["genres"].isnull().sum() / len(df) < 0.05  # <5% missing
