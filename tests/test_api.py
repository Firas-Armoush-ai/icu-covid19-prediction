"""Contract tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import THRESHOLD, app

EXAMPLE_PATIENT = {
    "AGE_PERCENTIL": "70th",
    "GENDER": 0,
    "HTN": 1,
    "IMMUNOCOMPROMISED": 0,
    "ALBUMIN_MEDIAN": 0.263,
    "BE_ARTERIAL_MEDIAN": -1.0,
    "PCR_MEDIAN": -0.958,
    "UREA_MEDIAN": -0.880,
    "CREATININ_MEDIAN": -0.924,
}

LABELS = {1: "Flag for ICU review", 0: "Below review threshold"}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def predict(client, **changes):
    response = client.post("/predict", json={**EXAMPLE_PATIENT, **changes})
    assert response.status_code == 200, response.text
    return response.json()


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["model"] == "random_forest+sigmoid"


def test_predict_returns_valid_response(client):
    body = predict(client)

    assert set(body) == {"icu_probability", "prediction", "risk_level", "threshold"}
    assert 0.0 <= body["icu_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] == LABELS[body["prediction"]]


def test_label_uses_saved_threshold(client):
    body = predict(client)

    assert body["threshold"] == round(THRESHOLD, 4)
    assert body["prediction"] == int(body["icu_probability"] >= THRESHOLD)


def test_predict_accepts_empty_payload(client):
    # Every field is optional; missing values are imputed by the pipeline.
    response = client.post("/predict", json={})
    assert response.status_code == 200


def test_lab_input_changes_the_estimate(client):
    low = predict(client, UREA_MEDIAN=-1.0)["icu_probability"]
    high = predict(client, UREA_MEDIAN=1.0)["icu_probability"]
    assert low != high


def test_age_group_changes_the_estimate(client):
    young = predict(client, AGE_PERCENTIL="10th")["icu_probability"]
    old = predict(client, AGE_PERCENTIL="Above 90th")["icu_probability"]
    assert young != old


@pytest.mark.parametrize("changes", [
    {"HEART_RATE_MEDIAN": 0.1},          # vital signs are not model inputs
    {"AGE_ABOVE65": 1},                   # derived from AGE_PERCENTIL, not an input
    {"UREA_MEDIAN": 3.0},                 # outside the provider's -1..1 scale
    {"AGE_PERCENTIL": "65"},              # not one of the dataset's groups
    {"HTN": 2},                           # binary
    {"GENDER": "not-a-number"},
])
def test_predict_rejects_invalid_input(client, changes):
    response = client.post("/predict", json={**EXAMPLE_PATIENT, **changes})
    assert response.status_code == 422


def test_predict_is_deterministic(client):
    assert predict(client) == predict(client)


def test_model_info(client):
    body = client.get("/model-info").json()

    assert body["threshold"] == round(THRESHOLD, 4)
    assert body["recall_target"] == 0.9
    assert body["test"]["n_patients"] == sum(body["test"]["confusion"].values())
