"""Contract tests for the FastAPI /predict endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

EXAMPLE_PATIENT = {
    "AGE_ABOVE65": 1,
    "GENDER": 0,
    "HTN": 1,
    "IMMUNOCOMPROMISED": 0,
    "ALBUMIN_MEDIAN": 0.263,
    "BE_ARTERIAL_MEDIAN": -1.0,
    "PCR_MEDIAN": -0.958,
    "UREA_MEDIAN": -0.880,
    "CREATININ_MEDIAN": -0.924,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=EXAMPLE_PATIENT)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"icu_probability", "prediction", "risk_level"}
    assert 0.0 <= body["icu_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("High Risk", "Low Risk")


def test_predict_label_matches_probability(client):
    body = client.post("/predict", json=EXAMPLE_PATIENT).json()

    expected_label = 1 if body["icu_probability"] >= 0.5 else 0
    assert body["prediction"] == expected_label


def test_predict_accepts_empty_payload(client):
    # Every field is optional; missing values are imputed by the pipeline.
    response = client.post("/predict", json={})
    assert response.status_code == 200


def test_predict_rejects_wrong_type(client):
    response = client.post("/predict", json={"AGE_ABOVE65": "not-a-number"})
    assert response.status_code == 422


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=EXAMPLE_PATIENT).json()
    second = client.post("/predict", json=EXAMPLE_PATIENT).json()
    assert first == second
