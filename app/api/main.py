# app/api/main.py

import json
from pathlib import Path
from typing import Literal, Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.utils.inference import (
    AGE_GROUPS,
    age_above_65,
    build_inference_dataframe,
    expand_lab_values,
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ICU Prediction API",
    description=(
        "Estimate ICU admission risk from the first two hours of a COVID-19 "
        "admission. Educational and portfolio project, not a clinical tool. "
        "By Firas Armoush."
    ),
    version="2.0"
)


# --------------------------------------------------
# Load the trained model and its metadata
# --------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

METADATA = json.loads((MODEL_DIR / "model_metadata.json").read_text())

model = joblib.load(MODEL_DIR / METADATA["model_file"])

THRESHOLD = METADATA["threshold"]
FEATURE_COLUMNS = METADATA["feature_columns"]

print(f"Loaded model: {METADATA['model']} (threshold {THRESHOLD:.3f})")


# --------------------------------------------------
# Input schema
# --------------------------------------------------

AgeGroup = Literal[tuple(AGE_GROUPS)]

# The data provider scaled every numeric column to the range -1 to 1.
def scaled(description):
    return Field(default=None, ge=-1.0, le=1.0, description=description)


class PatientData(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    AGE_PERCENTIL: Optional[AgeGroup] = Field(
        default=None,
        description="Age percentile group, as defined by the data provider",
    )
    GENDER: Optional[Literal[0, 1]] = Field(
        default=None,
        description="Gender as coded 0/1 in the dataset (the provider does not say which is which)",
    )
    HTN: Optional[Literal[0, 1]] = Field(default=None, description="Hypertension (1 = yes)")
    IMMUNOCOMPROMISED: Optional[Literal[0, 1]] = Field(default=None, description="1 = yes")

    ALBUMIN_MEDIAN: Optional[float] = scaled("Albumin, scaled -1 to 1")
    BE_ARTERIAL_MEDIAN: Optional[float] = scaled("Arterial base excess, scaled -1 to 1")
    PCR_MEDIAN: Optional[float] = scaled("C-reactive protein (PCR), scaled -1 to 1")
    UREA_MEDIAN: Optional[float] = scaled("Urea, scaled -1 to 1")
    CREATININ_MEDIAN: Optional[float] = scaled("Creatinine, scaled -1 to 1")


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "ICU Prediction API is running",
        "model": METADATA["model"],
    }


@app.get("/model-info")
def model_info():
    """The decision threshold and how the model performed on held-out data."""

    test = METADATA["test"]

    return {
        "model": METADATA["model"],
        "threshold": round(THRESHOLD, 4),
        "recall_target": METADATA["recall_target"],
        "test": {
            "n_patients": test["n_patients"],
            "n_icu": test["n_icu"],
            "metrics": {k: round(v, 3) for k, v in test["metrics"].items()},
            "confusion": test["confusion"],
        },
    }


@app.post("/predict")
def predict(data: PatientData):

    user_input = data.model_dump()

    if user_input["AGE_PERCENTIL"] is not None:
        user_input["AGE_ABOVE65"] = age_above_65(user_input["AGE_PERCENTIL"])

    user_input = expand_lab_values(user_input, FEATURE_COLUMNS)

    input_df = build_inference_dataframe(user_input, FEATURE_COLUMNS)

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= THRESHOLD)

    return {
        "icu_probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": (
            "Flag for ICU review"
            if prediction == 1
            else "Below review threshold"
        ),
        "threshold": round(THRESHOLD, 4),
    }
