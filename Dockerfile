# ── Build stage: install dependencies ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage: minimal image ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY app/ app/
COPY models/ models/

# PYTHONPATH so "from src.config import ..." resolves correctly
ENV PYTHONPATH=/app
ENV MODEL_VERSION=1.0.0

EXPOSE 8000

# uvicorn with 2 workers; adjust based on available CPUs
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
