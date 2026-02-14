"""
Script to generate the training notebook (train.ipynb) programmatically.

Run: python scripts/generate_notebook.py
"""
import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "train.ipynb")

cells = [
    # Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Movie Recommendation Engine — Training Notebook\n",
            "\n",
            "This notebook trains a TruncatedSVD-based collaborative filtering model on the MovieLens ml-latest-small dataset.\n",
            "\n",
            "**Steps:**\n",
            "1. Dataset overview & EDA\n",
            "2. Baseline predictor (Global Mean)\n",
            "3. SVD with default parameters\n",
            "4. Hyperparameter tuning\n",
            "5. Results summary & export\n",
            "\n",
            "**Target:** ~22% MAE reduction vs baseline"
        ]
    },
    # Imports
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import json, pickle, time, os\n",
            "from scipy.sparse import csr_matrix\n",
            "from sklearn.decomposition import TruncatedSVD\n",
            "from sklearn.model_selection import KFold"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Load Data
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Load & Explore Data"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Load processed ratings\n",
            "ratings = pd.read_csv('../data/processed/all_ratings.csv')\n",
            "movies = pd.read_csv('../data/processed/movies_metadata.csv')\n",
            "\n",
            "print(f'Ratings: {len(ratings):,}')\n",
            "print(f'Users:   {ratings[\"userId\"].nunique()}')\n",
            "print(f'Movies:  {ratings[\"movieId\"].nunique()}')\n",
            "print(f'Sparsity: {1 - len(ratings) / (ratings[\"userId\"].nunique() * ratings[\"movieId\"].nunique()):.2%}')\n",
            "print()\n",
            "ratings.describe()"
        ],
        "execution_count": None,
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Rating distribution\n",
            "print('Rating Distribution:')\n",
            "print(ratings['rating'].value_counts().sort_index())\n",
            "print(f'\\nMean rating: {ratings[\"rating\"].mean():.3f}')\n",
            "print(f'Median rating: {ratings[\"rating\"].median()}')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Build Matrix
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Build User-Item Matrix"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "user_ids = sorted(ratings['userId'].unique())\n",
            "movie_ids = sorted(ratings['movieId'].unique())\n",
            "u2i = {uid: i for i, uid in enumerate(user_ids)}\n",
            "m2i = {mid: i for i, mid in enumerate(movie_ids)}\n",
            "\n",
            "rows = ratings['userId'].map(u2i).values\n",
            "cols = ratings['movieId'].map(m2i).values\n",
            "vals = ratings['rating'].values\n",
            "matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))\n",
            "print(f'User-Item matrix: {matrix.shape}')\n",
            "print(f'Non-zero entries: {matrix.nnz:,}')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Baseline
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Baseline Predictor\n",
            "\n",
            "Using global mean rating as the baseline predictor."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
            "baseline_maes = []\n",
            "for fold, (train_idx, test_idx) in enumerate(kf.split(ratings), 1):\n",
            "    train_df = ratings.iloc[train_idx]\n",
            "    test_df = ratings.iloc[test_idx]\n",
            "    global_mean = train_df['rating'].mean()\n",
            "    mae = np.mean(np.abs(test_df['rating'].values - global_mean))\n",
            "    baseline_maes.append(mae)\n",
            "    print(f'Fold {fold}: MAE = {mae:.4f}')\n",
            "\n",
            "baseline_mae = np.mean(baseline_maes)\n",
            "print(f'\\nBaseline MAE: {baseline_mae:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Default SVD
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. SVD with Default Parameters\n",
            "\n",
            "TruncatedSVD decomposes the rating matrix R ≈ U × Σ × V^T into low-rank user/item factor matrices."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def evaluate_svd_cv(df, n_components=100, n_iter=10):\n",
            "    kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
            "    maes = []\n",
            "    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):\n",
            "        train_df = df.iloc[train_idx]\n",
            "        test_df = df.iloc[test_idx]\n",
            "        mat, uids, mids, u2i_f, m2i_f = build_matrix_from_df(train_df)\n",
            "        dense = mat.toarray().astype(np.float64)\n",
            "        gm = train_df['rating'].mean()\n",
            "        mask = dense != 0\n",
            "        um = np.where(mask.sum(axis=1)>0, (dense*mask).sum(axis=1)/mask.sum(axis=1), gm)\n",
            "        centred = dense.copy()\n",
            "        for i in range(centred.shape[0]):\n",
            "            centred[i, mask[i]] -= um[i]\n",
            "        svd = TruncatedSVD(n_components=min(n_components, min(centred.shape)-1), n_iter=n_iter, random_state=42)\n",
            "        uf = svd.fit_transform(centred)\n",
            "        itf = svd.components_.T\n",
            "        true_r, pred_r = [], []\n",
            "        for _, row in test_df.iterrows():\n",
            "            uid, mid, r = int(row['userId']), int(row['movieId']), row['rating']\n",
            "            if uid not in u2i_f or mid not in m2i_f: continue\n",
            "            s = um[u2i_f[uid]] + np.dot(uf[u2i_f[uid]], itf[m2i_f[mid]])\n",
            "            true_r.append(r); pred_r.append(np.clip(s, 0.5, 5.0))\n",
            "        maes.append(np.mean(np.abs(np.array(true_r) - np.array(pred_r))))\n",
            "        print(f'  Fold {fold}: MAE = {maes[-1]:.4f}')\n",
            "    return np.mean(maes)\n",
            "\n",
            "def build_matrix_from_df(df):\n",
            "    uids = sorted(df['userId'].unique())\n",
            "    mids = sorted(df['movieId'].unique())\n",
            "    u2i_f = {u: i for i, u in enumerate(uids)}\n",
            "    m2i_f = {m: i for i, m in enumerate(mids)}\n",
            "    r = df['userId'].map(u2i_f).values\n",
            "    c = df['movieId'].map(m2i_f).values\n",
            "    v = df['rating'].values\n",
            "    return csr_matrix((v, (r, c)), shape=(len(uids), len(mids))), uids, mids, u2i_f, m2i_f\n",
            "\n",
            "default_mae = evaluate_svd_cv(ratings, n_components=100, n_iter=10)\n",
            "print(f'\\nDefault SVD MAE: {default_mae:.4f}')\n",
            "print(f'Improvement vs Baseline: {(baseline_mae - default_mae) / baseline_mae * 100:.1f}%')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Tuning
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Hyperparameter Tuning\n",
            "\n",
            "Searching over n_components and n_iter."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "best_mae = float('inf')\n",
            "best_params = {}\n",
            "for nc in [50, 100, 150, 200]:\n",
            "    for ni in [5, 10, 20]:\n",
            "        mae = evaluate_svd_cv(ratings, n_components=nc, n_iter=ni)\n",
            "        if mae < best_mae:\n",
            "            best_mae = mae\n",
            "            best_params = {'n_components': nc, 'n_iter': ni}\n",
            "            print(f'  New best: nc={nc}, ni={ni} -> MAE={mae:.4f}')\n",
            "\n",
            "tuned_mae = best_mae\n",
            "print(f'\\nBest MAE:    {tuned_mae:.4f}')\n",
            "print(f'Best params: {json.dumps(best_params, indent=2)}')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Summary
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Results Summary"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "improvement = (baseline_mae - tuned_mae) / baseline_mae * 100\n",
            "\n",
            "print('=' * 50)\n",
            "print('FINAL RESULTS')\n",
            "print('=' * 50)\n",
            "print(f'{\"Baseline MAE (Global Mean)\":<30} {baseline_mae:.4f}')\n",
            "print(f'{\"Default SVD MAE\":<30} {default_mae:.4f}')\n",
            "print(f'{\"Tuned SVD MAE\":<30} {tuned_mae:.4f}')\n",
            "print(f'{\"MAE reduction vs Baseline\":<30} {improvement:.1f}%')\n",
            "print()\n",
            "print('Best Hyperparameters:')\n",
            "for k, v in best_params.items():\n",
            "    print(f'  {k}: {v}')\n",
            "\n",
            "if improvement >= 20:\n",
            "    print(f'\\n✓ TARGET MET: {improvement:.1f}% MAE reduction')\n",
            "else:\n",
            "    print(f'\\n⚠ Achieved {improvement:.1f}% (target ~22%)')"
        ],
        "execution_count": None,
        "outputs": []
    },
    # Export
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Train Final Model & Export"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Train on full dataset with best params\n",
            "mat, uids, mids, u2i_f, m2i_f = build_matrix_from_df(ratings)\n",
            "dense = mat.toarray().astype(np.float64)\n",
            "gm = ratings['rating'].mean()\n",
            "mask = dense != 0\n",
            "um = np.where(mask.sum(axis=1)>0, (dense*mask).sum(axis=1)/mask.sum(axis=1), gm)\n",
            "centred = dense.copy()\n",
            "for i in range(centred.shape[0]):\n",
            "    centred[i, mask[i]] -= um[i]\n",
            "\n",
            "nc = best_params['n_components']\n",
            "ni = best_params['n_iter']\n",
            "svd = TruncatedSVD(n_components=min(nc, min(centred.shape)-1), n_iter=ni, random_state=42)\n",
            "uf = svd.fit_transform(centred)\n",
            "itf = svd.components_.T\n",
            "\n",
            "user_rated = {}\n",
            "for uid in uids:\n",
            "    u_idx = u2i_f[uid]\n",
            "    rated_indices = mat[u_idx].nonzero()[1]\n",
            "    user_rated[uid] = {mids[j] for j in rated_indices}\n",
            "\n",
            "os.makedirs('../models', exist_ok=True)\n",
            "with open('../models/svd_model.pkl', 'wb') as f:\n",
            "    pickle.dump({\n",
            "        'user_factors': uf, 'item_factors': itf,\n",
            "        'global_mean': gm, 'user_means': um,\n",
            "        'user_ids': uids, 'movie_ids': mids,\n",
            "        'user_to_idx': u2i_f, 'movie_to_idx': m2i_f,\n",
            "        'user_rated': user_rated, 'best_params': best_params,\n",
            "    }, f)\n",
            "\n",
            "metrics = {\n",
            "    'baseline_mae': round(baseline_mae, 4),\n",
            "    'default_svd_mae': round(default_mae, 4),\n",
            "    'tuned_svd_mae': round(tuned_mae, 4),\n",
            "    'mae_reduction_pct': round(improvement, 1),\n",
            "    'best_params': best_params,\n",
            "    'dataset': 'MovieLens ml-latest-small',\n",
            "    'algorithm': 'TruncatedSVD (Matrix Factorization)',\n",
            "    'library': 'scikit-learn',\n",
            "}\n",
            "with open('../models/metrics.json', 'w') as f:\n",
            "    json.dump(metrics, f, indent=2)\n",
            "\n",
            "print('Model and metrics saved successfully.')"
        ],
        "execution_count": None,
        "outputs": []
    },
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "cells": cells,
}

os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
with open(NOTEBOOK_PATH, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created at {NOTEBOOK_PATH}")
