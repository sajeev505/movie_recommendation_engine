"""
Train SVD collaborative filtering model on MovieLens ml-latest-small.

This script:
1. Loads preprocessed ratings data
2. Builds a User-Item matrix
3. Computes a global-mean baseline MAE
4. Trains TruncatedSVD models with cross-validation
5. Performs hyperparameter search to optimize MAE
6. Reports MAE improvement vs baseline (~22% target)
7. Exports the best model artefacts to models/svd_model.pkl

Usage:
    python scripts/train_model.py
"""

import os
import sys
import json
import time
import pickle

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def load_data():
    """Load ratings CSV and build user-item matrix."""
    ratings_path = os.path.join(PROCESSED_DIR, "all_ratings.csv")
    if not os.path.exists(ratings_path):
        print("ERROR: Run scripts/preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(ratings_path)
    return df


def build_matrix(df):
    """Build a sparse User×Movie rating matrix and return index mappings."""
    user_ids = sorted(df["userId"].unique())
    movie_ids = sorted(df["movieId"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    rows = df["userId"].map(user_to_idx).values
    cols = df["movieId"].map(movie_to_idx).values
    vals = df["rating"].values

    matrix = csr_matrix((vals, (rows, cols)),
                        shape=(len(user_ids), len(movie_ids)))
    return matrix, user_ids, movie_ids, user_to_idx, movie_to_idx


def compute_mae(true_ratings, predicted_ratings):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(true_ratings - predicted_ratings))


def predict_ratings(user_factors, item_factors, global_mean, user_means,
                    user_indices, item_indices):
    """Predict ratings for specific (user, item) pairs."""
    predictions = []
    for u_idx, i_idx in zip(user_indices, item_indices):
        score = global_mean + (user_means[u_idx] - global_mean) + \
                np.dot(user_factors[u_idx], item_factors[i_idx])
        # Clip to valid rating range
        score = np.clip(score, 0.5, 5.0)
        predictions.append(score)
    return np.array(predictions)


def evaluate_baseline(df):
    """Compute baseline MAE using global mean predictor (5-fold CV)."""
    print("\n" + "=" * 60)
    print("STEP 1: Baseline Predictor (Global Mean)")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        global_mean = train_df["rating"].mean()
        mae = np.mean(np.abs(test_df["rating"].values - global_mean))
        maes.append(mae)
        print(f"  Fold {fold}: MAE = {mae:.4f}")

    mean_mae = np.mean(maes)
    print(f"\nBaseline MAE:  {mean_mae:.4f}")
    return mean_mae


def evaluate_svd(df, n_components=100, n_iter=10, label="Default SVD"):
    """Train TruncatedSVD and evaluate with 5-fold CV. Returns mean MAE."""
    print(f"\n{'=' * 60}")
    print(f"{label} (n_components={n_components}, n_iter={n_iter})")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Build matrix from training fold
        matrix, user_ids, movie_ids, u2i, m2i = build_matrix(train_df)
        dense = matrix.toarray().astype(np.float64)
        global_mean = train_df["rating"].mean()

        # Centre the matrix (subtract global mean from non-zero entries)
        mask = dense != 0
        user_means = np.where(mask.sum(axis=1) > 0,
                              (dense * mask).sum(axis=1) / mask.sum(axis=1),
                              global_mean)
        centred = dense.copy()
        for i in range(centred.shape[0]):
            centred[i, mask[i]] -= user_means[i]

        # Fit SVD
        svd = TruncatedSVD(n_components=min(n_components, min(centred.shape) - 1),
                           n_iter=n_iter, random_state=42)
        user_factors = svd.fit_transform(centred)
        item_factors = svd.components_.T  # (n_movies, n_components)

        # Predict on test fold
        test_true = []
        test_pred = []
        for _, row in test_df.iterrows():
            uid, mid, rating = int(row["userId"]), int(row["movieId"]), row["rating"]
            if uid not in u2i or mid not in m2i:
                continue
            u_idx = u2i[uid]
            i_idx = m2i[mid]
            score = user_means[u_idx] + np.dot(user_factors[u_idx], item_factors[i_idx])
            score = np.clip(score, 0.5, 5.0)
            test_true.append(rating)
            test_pred.append(score)

        mae = compute_mae(np.array(test_true), np.array(test_pred))
        maes.append(mae)
        print(f"  Fold {fold}: MAE = {mae:.4f}")

    mean_mae = np.mean(maes)
    print(f"\n{label} MAE: {mean_mae:.4f}")
    return mean_mae


def tune_svd(df):
    """Search over hyperparameters. Returns best MAE and params."""
    print("\n" + "=" * 60)
    print("STEP 3: Hyperparameter Tuning")
    print("=" * 60)

    param_grid = [
        {"n_components": nc, "n_iter": ni}
        for nc in [50, 100, 150, 200]
        for ni in [5, 10, 20]
    ]

    print(f"Searching {len(param_grid)} parameter combinations (3-fold CV)...")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_mae = float("inf")
    best_params = {}

    for params in param_grid:
        nc, ni = params["n_components"], params["n_iter"]
        maes = []

        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            matrix, user_ids, movie_ids, u2i, m2i = build_matrix(train_df)
            dense = matrix.toarray().astype(np.float64)
            global_mean = train_df["rating"].mean()

            mask = dense != 0
            user_means = np.where(mask.sum(axis=1) > 0,
                                  (dense * mask).sum(axis=1) / mask.sum(axis=1),
                                  global_mean)
            centred = dense.copy()
            for i in range(centred.shape[0]):
                centred[i, mask[i]] -= user_means[i]

            svd = TruncatedSVD(n_components=min(nc, min(centred.shape) - 1),
                               n_iter=ni, random_state=42)
            user_factors = svd.fit_transform(centred)
            item_factors = svd.components_.T

            test_true, test_pred = [], []
            for _, row in test_df.iterrows():
                uid, mid, rating = int(row["userId"]), int(row["movieId"]), row["rating"]
                if uid not in u2i or mid not in m2i:
                    continue
                u_idx, i_idx = u2i[uid], m2i[mid]
                score = user_means[u_idx] + np.dot(user_factors[u_idx], item_factors[i_idx])
                score = np.clip(score, 0.5, 5.0)
                test_true.append(rating)
                test_pred.append(score)

            mae = compute_mae(np.array(test_true), np.array(test_pred))
            maes.append(mae)

        avg_mae = np.mean(maes)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
            print(f"  New best: n_components={nc}, n_iter={ni} → MAE={avg_mae:.4f}")

    print(f"\nBest MAE:    {best_mae:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    return best_mae, best_params


def train_final_model(df, best_params):
    """Train final SVD on full dataset with best params. Save to disk."""
    print("\n" + "=" * 60)
    print("STEP 4: Training Final Model on Full Dataset")
    print("=" * 60)

    matrix, user_ids, movie_ids, u2i, m2i = build_matrix(df)
    dense = matrix.toarray().astype(np.float64)
    global_mean = df["rating"].mean()

    mask = dense != 0
    user_means = np.where(mask.sum(axis=1) > 0,
                          (dense * mask).sum(axis=1) / mask.sum(axis=1),
                          global_mean)
    centred = dense.copy()
    for i in range(centred.shape[0]):
        centred[i, mask[i]] -= user_means[i]

    nc = best_params["n_components"]
    ni = best_params["n_iter"]

    svd = TruncatedSVD(n_components=min(nc, min(centred.shape) - 1),
                       n_iter=ni, random_state=42)
    start = time.time()
    user_factors = svd.fit_transform(centred)
    item_factors = svd.components_.T
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.2f}s")
    print(f"Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

    # Build user_rated mapping: {raw_user_id: set(raw_movie_ids)}
    user_rated = {}
    for uid in user_ids:
        u_idx = u2i[uid]
        rated_indices = matrix[u_idx].nonzero()[1]
        user_rated[uid] = {movie_ids[j] for j in rated_indices}

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "svd_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "user_factors": user_factors,
            "item_factors": item_factors,
            "global_mean": global_mean,
            "user_means": user_means,
            "user_ids": user_ids,
            "movie_ids": movie_ids,
            "user_to_idx": u2i,
            "movie_to_idx": m2i,
            "user_rated": user_rated,
            "best_params": best_params,
        }, f)

    print(f"Model saved to {model_path}")
    print(f"  Users: {len(user_ids)}, Movies: {len(movie_ids)}")
    print(f"  Latent factors: {user_factors.shape[1]}")


def print_summary(baseline_mae, default_mae, tuned_mae, best_params):
    """Print final summary with % improvement."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    improvement_vs_baseline = (baseline_mae - tuned_mae) / baseline_mae * 100
    improvement_vs_default = (default_mae - tuned_mae) / default_mae * 100

    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Baseline MAE (Global Mean)':<30} {baseline_mae:>10.4f}")
    print(f"{'Default SVD MAE':<30} {default_mae:>10.4f}")
    print(f"{'Tuned SVD MAE':<30} {tuned_mae:>10.4f}")
    print(f"{'MAE reduction vs Baseline':<30} {improvement_vs_baseline:>9.1f}%")
    print(f"{'MAE reduction vs Default SVD':<30} {improvement_vs_default:>9.1f}%")
    print()

    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    if improvement_vs_baseline >= 20:
        print(f"\n✓ Target achieved: {improvement_vs_baseline:.1f}% MAE reduction vs baseline (target ≈ 22%)")
    else:
        print(f"\n⚠ Achieved {improvement_vs_baseline:.1f}% MAE reduction vs baseline (target ≈ 22%)")
        print("  Consider increasing n_components or n_iter.")

    return improvement_vs_baseline


def save_metrics(baseline_mae, default_mae, tuned_mae, best_params, pct_improvement):
    """Save metrics to JSON for the API to serve."""
    metrics = {
        "baseline_mae": round(baseline_mae, 4),
        "default_svd_mae": round(default_mae, 4),
        "tuned_svd_mae": round(tuned_mae, 4),
        "mae_reduction_pct": round(pct_improvement, 1),
        "best_params": best_params,
        "dataset": "MovieLens ml-latest-small",
        "algorithm": "TruncatedSVD (Matrix Factorization)",
        "library": "scikit-learn",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")


def main():
    print("=" * 60)
    print("MOVIE RECOMMENDATION ENGINE — MODEL TRAINING")
    print("Dataset: MovieLens ml-latest-small (100K ratings)")
    print("Library: scikit-learn TruncatedSVD")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded {len(df):,} ratings from {df['userId'].nunique()} users "
          f"on {df['movieId'].nunique()} movies")

    # Step 1: Baseline
    baseline_mae = evaluate_baseline(df)

    # Step 2: Default SVD
    default_mae = evaluate_svd(df, n_components=100, n_iter=10,
                               label="STEP 2: Default SVD")

    # Step 3: Hyperparameter tuning
    tuned_mae, best_params = tune_svd(df)

    # Step 4: Train final model
    train_final_model(df, best_params)

    # Step 5: Report
    pct_improvement = print_summary(baseline_mae, default_mae, tuned_mae, best_params)
    save_metrics(baseline_mae, default_mae, tuned_mae, best_params, pct_improvement)


if __name__ == "__main__":
    main()
