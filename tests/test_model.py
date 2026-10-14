"""The saved model loads and produces valid probabilities."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.api import main as api
from src.config import DATA_PATH, ID_COL, TARGET_COL


@pytest.fixture(scope="module")
def model():
    return api.model


def test_model_is_fitted_pipeline(model):
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict_proba")
    assert len(model.feature_names_in_) > 0


def test_model_expects_no_target_or_id(model):
    features = set(model.feature_names_in_)
    assert TARGET_COL not in features
    assert ID_COL not in features


def test_model_predicts_on_training_schema(model):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL, ID_COL]).head(10)

    proba = model.predict_proba(X[model.feature_names_in_])

    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
