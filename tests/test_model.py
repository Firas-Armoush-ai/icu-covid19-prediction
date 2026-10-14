"""The saved model and its metadata load, agree, and produce valid probabilities."""

import json

import joblib
import pandas as pd
import pytest

from src.config import DATA_PATH, FINAL_MODEL, ID_COL, METADATA_PATH, MODEL_PATH, RECALL_TARGET, TARGET_COL


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text())


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_metadata_matches_config(metadata):
    assert metadata["model"] == FINAL_MODEL
    assert metadata["recall_target"] == RECALL_TARGET
    assert metadata["model_file"] == MODEL_PATH.name
    assert 0.0 < metadata["threshold"] < 1.0


def test_model_columns_match_metadata(model, metadata):
    assert list(model.feature_names_in_) == metadata["feature_columns"]


def test_model_expects_no_target_or_id(metadata):
    assert TARGET_COL not in metadata["feature_columns"]
    assert ID_COL not in metadata["feature_columns"]


def test_model_predicts_on_training_schema(model, metadata):
    df = pd.read_csv(DATA_PATH)
    X = df[metadata["feature_columns"]].head(10)

    proba = model.predict_proba(X)

    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_test_results_are_consistent(metadata):
    test = metadata["test"]
    c = test["confusion"]

    assert c["tp"] + c["fn"] == test["n_icu"]
    assert sum(c.values()) == test["n_patients"]
    assert test["metrics"]["recall"] == pytest.approx(c["tp"] / test["n_icu"])
