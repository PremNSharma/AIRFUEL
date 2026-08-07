# ============================================================
# Aircraft Fuel Consumption Model - Multi-stage Dockerfile
# ============================================================

# ── Base Python image ──────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies stage ─────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Backend stage ──────────────────────────────────────────
FROM dependencies AS backend

COPY backend/ ./backend/
COPY config/ ./config/
COPY database/ ./database/
COPY models/ ./models/
COPY training/ ./training/
COPY utils/ ./utils/
COPY services/ ./services/
COPY analytics/ ./analytics/
COPY .env.example ./.env

RUN mkdir -p data experiments/mlruns experiments/artifacts models logs

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ── Training stage ─────────────────────────────────────────
FROM dependencies AS trainer

COPY training/ ./training/
COPY config/ ./config/
COPY utils/ ./utils/
COPY data/ ./data/

CMD ["python", "-m", "training.train_pipeline"]

# ── Test stage ─────────────────────────────────────────────
FROM dependencies AS test

COPY . .
RUN pip install pytest pytest-asyncio pytest-cov httpx

CMD ["pytest", "tests/", "--cov=backend", "--cov-report=xml", "-v"]
