# Deployment Guide

## Table of Contents
1. [Local Development (without Docker)](#local-development)
2. [Local Development (with Docker)](#docker-local)
3. [OAuth Setup](#oauth-setup)
4. [Deploy to Free Hosts](#deploy-to-free-hosts)
5. [Environment Variables](#environment-variables)

---

## 1. Local Development (without Docker) <a name="local-development"></a>

### Prerequisites
- Python 3.13+ (required — scikit-learn and numpy versions target this)
- pip

### Steps

```bash
# Create and activate virtual environment
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Preprocess data
python scripts/preprocess.py

# Train model (takes 2-5 minutes depending on CPU)
# Uses scikit-learn TruncatedSVD with hyperparameter search
python scripts/train_model.py

# Copy environment file
cp .env.example .env

# Start API server
python -m src.backend.app
# API running at http://localhost:5000

# Open frontend in browser
# Open src/frontend/index.html directly, or use a local server:
python -m http.server 3000 --directory src/frontend
# Frontend at http://localhost:3000
```

### Demo Mode (Default)
By default, `DEMO_MODE=true` is set. This bypasses OAuth — click "Try Demo" on
the frontend to explore recommendations without any GitHub app registration.

---

## 2. Local Development (with Docker) <a name="docker-local"></a>

### Prerequisites
- Docker and Docker Compose

### Steps

```bash
# First: download data and train model locally (not done inside Docker)
python scripts/download_data.py
python scripts/preprocess.py
python scripts/train_model.py

# Then build and start containers
docker-compose up --build

# Frontend: http://localhost:3000
# API:      http://localhost:5000
```

To stop:
```bash
docker-compose down
```

---

## 3. OAuth Setup (Optional) <a name="oauth-setup"></a>

To enable GitHub OAuth sign-in:

1. Go to https://github.com/settings/developers
2. Click **New OAuth App**
3. Fill in:
   - **Application name**: Movie Recommender
   - **Homepage URL**: http://localhost:3000
   - **Authorization callback URL**: http://localhost:5000/auth/github/callback
4. Click **Register application**
5. Copy the **Client ID** and generate a **Client Secret**
6. Set environment variables:

```bash
export GITHUB_CLIENT_ID=your_client_id_here
export GITHUB_CLIENT_SECRET=your_client_secret_here
export DEMO_MODE=false
```

Or add to `.env`:
```
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here
DEMO_MODE=false
```

---

## 4. Deploy to Free Hosts <a name="deploy-to-free-hosts"></a>

Recommended combo: **Render** (backend) + **Vercel** (frontend).

### Backend → Render

1. Push repo to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt && python scripts/download_data.py && python scripts/preprocess.py && python scripts/train_model.py`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 src.backend.app:app`
   - **Environment**: Python 3 (set to **3.13** in Render dashboard)
   - **Note**: numpy 2.2.3 and scikit-learn 1.6.1 require Python 3.13+
5. Add environment variables:
   - `FLASK_SECRET_KEY` = (a random string)
   - `DEMO_MODE` = `true`
   - `FRONTEND_URL` = `https://your-app.vercel.app`
6. Deploy

### Frontend → Vercel

1. Go to https://vercel.com → New Project
2. Import your GitHub repo
3. Settings:
   - **Root Directory**: `src/frontend`
   - **Framework Preset**: Other
   - **Build Command**: (leave empty)
   - **Output Directory**: `.`
4. Update `app.js` line 7: set `API_URL` to your Render backend URL
5. Deploy

### Update OAuth Callback (if using OAuth)

After deploying, update your GitHub OAuth App's callback URL to:
```
https://your-render-app.onrender.com/auth/github/callback
```

---

## 5. Environment Variables <a name="environment-variables"></a>

| Variable | Required | Default | Description |
|---|---|---|---|
| `FLASK_SECRET_KEY` | Yes (production) | `dev-secret-change-me` | Flask session secret |
| `DEMO_MODE` | No | `true` | Set `false` to require OAuth |
| `GITHUB_CLIENT_ID` | Only if OAuth | empty | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | Only if OAuth | empty | GitHub OAuth app secret |
| `MODEL_PATH` | No | `models/svd_model.pkl` | Path to trained model |
| `MOVIES_PATH` | No | `data/processed/movies_metadata.csv` | Path to movie metadata |
| `METRICS_PATH` | No | `models/metrics.json` | Path to metrics JSON |
| `FRONTEND_URL` | No | `http://localhost:3000` | Frontend origin for CORS |
| `API_URL` | No | `http://localhost:5000` | API base URL |

---

## Git Commands to Push

```bash
cd movie_recommendation_engine
git init
git add .
git commit -m "Initial commit: Movie Recommendation Engine with scikit-learn TruncatedSVD collaborative filtering"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/movie-recommendation-engine.git
git push -u origin main
```
