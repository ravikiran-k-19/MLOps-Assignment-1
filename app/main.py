"""
FastAPI prediction server for Heart Disease classification.

Endpoints:
  GET  /health   — liveness / readiness probe
  POST /predict  — predict heart disease from patient features (JSON in, JSON out)
  GET  /metrics  — Prometheus metrics scrape endpoint

Run locally:
  uvicorn app.main:app --reload --port 8000

API docs (auto-generated Swagger):
  http://localhost:8000/docs
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pythonjsonlogger import jsonlogger

from app.schemas import PatientFeatures, PredictionResponse
from src.config import CATEGORICAL_FEATURES, MODELS_DIR, NUMERIC_FEATURES

# ── Structured JSON logging ───────────────────────────────────────────────────
logger = logging.getLogger("heart_disease_api")
_handler = logging.StreamHandler()
_handler.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total predictions made",
    ["result"],  # "0" or "1"
)

# ── Model state ───────────────────────────────────────────────────────────────
_model = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup; release on shutdown."""
    global _model
    model_path = MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        _model = joblib.load(model_path)
        logger.info("model_loaded", extra={"path": str(model_path), "version": MODEL_VERSION})
    else:
        logger.warning(
            "model_file_missing",
            extra={"path": str(model_path), "hint": "run `python -m src.train` first"},
        )
    yield
    _model = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment 01 — Binary classifier for the Heart Disease UCI dataset",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Middleware: metrics + request logging ─────────────────────────────────────

@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - t0

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": endpoint,
            "status": response.status_code,
            "duration_s": round(duration, 4),
        },
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"], summary="Liveness probe")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": MODEL_VERSION,
    }


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientFeatures):
    """
    Predict heart disease from patient clinical features.

    Returns:
    - **prediction**: 0 (no disease) or 1 (disease)
    - **confidence**: probability of heart disease
    - **model_version**: deployed model version
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` to train and save a model.",
        )

    # Build a DataFrame in the exact column order expected by the pipeline
    features = patient.model_dump()
    df = pd.DataFrame([features])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    proba = float(_model.predict_proba(df)[:, 1][0])
    label = int(proba >= 0.5)

    PREDICTION_COUNT.labels(result=str(label)).inc()
    logger.info("prediction", extra={"prediction": label, "confidence": round(proba, 4)})

    return PredictionResponse(
        prediction=label,
        confidence=round(proba, 4),
        model_version=MODEL_VERSION,
    )
