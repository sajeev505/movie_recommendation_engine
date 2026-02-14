"""Preprocess MovieLens data: clean, merge, and split into train/test."""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-latest-small")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def preprocess():
    """Load raw CSVs, clean, merge movie metadata, and create train/test split."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Load raw data ---
    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    links = pd.read_csv(os.path.join(RAW_DIR, "links.csv"))

    print(f"Ratings: {len(ratings):,} rows")
    print(f"Movies:  {len(movies):,} rows")
    print(f"Links:   {len(links):,} rows")

    # --- Clean movies: extract year from title ---
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    # --- Merge links for TMDB poster IDs ---
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    movies["tmdbId"] = movies["tmdbId"].fillna(0).astype(int)

    # --- Save movie metadata lookup ---
    movies.to_csv(os.path.join(PROCESSED_DIR, "movies_metadata.csv"), index=False)
    print(f"Saved movies_metadata.csv ({len(movies)} movies)")

    # --- Train/test split (80/20 by timestamp) ---
    ratings = ratings.sort_values("timestamp")
    split_idx = int(len(ratings) * 0.8)
    train = ratings.iloc[:split_idx]
    test = ratings.iloc[split_idx:]

    train.to_csv(os.path.join(PROCESSED_DIR, "train_ratings.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test_ratings.csv"), index=False)
    ratings.to_csv(os.path.join(PROCESSED_DIR, "all_ratings.csv"), index=False)

    print(f"Train: {len(train):,} ratings")
    print(f"Test:  {len(test):,} ratings")
    print("Preprocessing complete.")

    # --- Summary statistics ---
    print("\n--- Dataset Summary ---")
    print(f"Users:  {ratings['userId'].nunique()}")
    print(f"Movies: {ratings['movieId'].nunique()}")
    print(f"Rating range: {ratings['rating'].min()} – {ratings['rating'].max()}")
    print(f"Mean rating:  {ratings['rating'].mean():.3f}")
    print(f"Sparsity:     {1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.4%}")


if __name__ == "__main__":
    preprocess()
