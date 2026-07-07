"""
Unit tests for app/main.py (FastAPI endpoints).

The model is mocked so these tests run without a trained model file.
We patch `app.main._model` directly after the app is imported.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import app.main as main_module
from app.main import app

# ── Sample payload matching PatientFeatures schema ────────────────────────────
VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Inject a mock sklearn pipeline that returns a fixed positive prediction."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.25, 0.75]])
    original = main_module._model
    main_module._model = mock
    yield mock
    main_module._model = original


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body_has_status(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200_with_mock(client, mock_model):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200


def test_predict_response_has_required_fields(client, mock_model):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert "prediction" in body
    assert "confidence" in body
    assert "model_version" in body


def test_predict_prediction_is_binary(client, mock_model):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert body["prediction"] in (0, 1)


def test_predict_confidence_in_range(client, mock_model):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_positive_case(client, mock_model):
    """Mock returns proba=0.75, so label should be 1."""
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert body["prediction"] == 1
    assert body["confidence"] == pytest.approx(0.75)


def test_predict_missing_field_returns_422(client, mock_model):
    """Schema validation must reject requests missing required features."""
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_503_when_model_not_loaded(client):
    """When _model is None the API must return 503."""
    original = main_module._model
    main_module._model = None
    response = client.post("/predict", json=VALID_PATIENT)
    main_module._model = original
    assert response.status_code == 503
