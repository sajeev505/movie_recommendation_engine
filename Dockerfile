# Backend
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/backend/ src/backend/
COPY src/__init__.py src/__init__.py
COPY models/ models/
COPY data/processed/ data/processed/

# Environment defaults
ENV FLASK_SECRET_KEY=docker-secret-change-me
ENV DEMO_MODE=true
ENV MODEL_PATH=/app/models/svd_model.pkl
ENV METRICS_PATH=/app/models/metrics.json
ENV MOVIES_PATH=/app/data/processed/movies_metadata.csv
ENV FRONTEND_URL=http://localhost:3000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "src.backend.app:app"]
