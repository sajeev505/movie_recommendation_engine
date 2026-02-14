# Personalized Movie Recommendation Engine

> **CV Bullet**: Engineered collaborative filtering recommendation engine using Matrix Factorization to personalize content discovery for platform users. Optimized Singular Value Decomposition algorithm parameters, reducing Mean Absolute Error by ~22% against standard baseline benchmarks. Deployed RESTful API using Flask to serve real-time movie suggestions achieving consistent sub-100ms inference latency.

---

## Quick Summary

| Metric | Value |
|---|---|
| **Algorithm** | SVD (Matrix Factorization) via scikit-learn |
| **Dataset** | MovieLens ml-latest-small (100K ratings, 600 users, 9K movies) |
| **Baseline MAE** | ~0.87 (Global Mean) |
| **Tuned SVD MAE** | ~0.68 |
| **MAE Reduction** | ~22% vs baseline |
| **Inference Latency** | < 100ms median (single request) |
| **API Framework** | Flask + Gunicorn |
| **Frontend** | Vanilla HTML/CSS/JS (dark theme, responsive) |

---

## 1-Minute Quickstart

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd movie_recommendation_engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data, preprocess, and train
python scripts/download_data.py
python scripts/preprocess.py
python scripts/train_model.py

# 5. Start the API
python -m src.backend.app

# 6. Open the frontend
# Open src/frontend/index.html in your browser
# Or use Docker (see below)
```

### Docker Quickstart

```bash
# Ensure data is downloaded and model is trained first (steps 4 above)
docker-compose up --build
# Frontend: http://localhost:3000
# API:      http://localhost:5000
```

---

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│   Frontend SPA  │ ────────────► │  Flask REST API   │
│  (HTML/CSS/JS)  │  localhost:3000│  localhost:5000   │
│  nginx / static │ ◄──────────── │  Gunicorn         │
└─────────────────┘    JSON       └────────┬─────────┘
                                           │
                                  ┌────────▼─────────┐
                                  │  SVD Model (.pkl) │
                                  │  Movie Metadata   │
                                  │  scikit-learn      │
                                  └──────────────────┘
```

---

## Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Dataset | MovieLens ml-latest-small | 100K ratings — fast to train, standard benchmark, free |
| Algorithm | scikit-learn TruncatedSVD | Battle-tested MF implementation, Python 3.13 compatible, reproducible |
| Baseline | Global Mean | Simple but effective baseline for collaborative filtering |
| Backend | Flask | Lightweight, matches CV claim, fast prototyping |
| Frontend | Vanilla HTML/CSS/JS | No build step, zero dependencies, serves from nginx |
| Poster images | Genre-based emoji placeholders (CC0) | No external image API required, fully offline |
| Free hosting | Render (backend) + Vercel (frontend) | Generous free tiers, supports env vars, easy deploy |

---

## Project Structure

```
├── README.md
├── DEPLOYMENT.md
├── BENCHMARKS.txt
├── ARCHITECTURE.txt
├── SAMPLE_REQUESTS.txt
├── RELEASE_NOTES.txt
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── Dockerfile             # Backend
├── Dockerfile.frontend    # Frontend (nginx)
├── docker-compose.yml
├── nginx.conf
├── .github/workflows/ci.yml
├── scripts/
│   ├── download_data.py   # Download MovieLens dataset
│   ├── preprocess.py      # Clean and split data
│   ├── train_model.py     # Train SVD model with tuning
│   └── benchmark.py       # Inference latency benchmark
├── notebooks/
│   └── train.ipynb        # Training notebook with EDA
├── models/                # Generated model artifacts (gitignored)
├── data/                  # Raw & processed data (gitignored)
├── src/
│   ├── backend/
│   │   ├── app.py         # Flask application factory
│   │   ├── config.py      # Environment configuration
│   │   ├── recommender.py # SVD recommendation engine
│   │   ├── routes.py      # API endpoints
│   │   └── auth.py        # GitHub OAuth + demo mode
│   └── frontend/
│       ├── index.html     # Single-page app
│       ├── styles.css     # Design system
│       └── app.js         # Client logic
└── tests/
    ├── test_recommender.py
    ├── test_api.py
    └── test_data_pipeline.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/recommendations/<user_id>?n=10` | Top-N recommendations |
| GET | `/api/movies/search?q=<query>` | Search movies by title |
| GET | `/api/user/<user_id>/ratings` | User's rated movies |
| GET | `/api/users` | Sample user IDs |
| GET | `/auth/demo` | Demo login (no OAuth) |
| GET | `/auth/github` | GitHub OAuth login |
| GET | `/auth/me` | Current user info |
| GET | `/auth/logout` | Logout |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Running Benchmark

```bash
python scripts/benchmark.py
```

## License

MIT — see [LICENSE](LICENSE).
