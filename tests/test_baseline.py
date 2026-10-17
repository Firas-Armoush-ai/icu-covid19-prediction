"""The age-only baseline and the paired bootstrap used to compare against it."""

import numpy as np
import pytest

from src.config import AGE_GROUPS, BASELINE_MODEL
from src.evaluate import paired_bootstrap_difference
from src.models import build_model, load_split


@pytest.fixture(scope="module")
def fitted_baseline():
    X_train, X_test, y_train, _ = load_split()
    return build_model(BASELINE_MODEL).fit(X_train, y_train), X_test


def test_baseline_ignores_everything_but_age(fitted_baseline):
    model, X_test = fitted_baseline
    changed = X_test.copy()
    for col in changed.columns:
        if col != "AGE_PERCENTIL" and changed[col].dtype.kind == "f":
            changed[col] = 1.0

    assert np.allclose(model.predict_proba(X_test), model.predict_proba(changed))


def test_baseline_risk_rises_with_age_group(fitted_baseline):
    model, X_test = fitted_baseline
    rows = X_test.iloc[[0] * len(AGE_GROUPS)].copy()
    rows["AGE_PERCENTIL"] = AGE_GROUPS

    risk = model.predict_proba(rows)[:, 1]
    assert np.all(np.diff(risk) > 0)


def test_paired_difference_is_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    p = rng.random(60)

    result = paired_bootstrap_difference(y, p, p, n_bootstrap=200)
    assert result == {"difference": 0.0, "low": 0.0, "high": 0.0}


def test_paired_difference_favours_the_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good = y * 0.6 + rng.random(200) * 0.4  # separates the classes well
    noise = rng.random(200)

    result = paired_bootstrap_difference(y, good, noise, n_bootstrap=200)
    assert result["low"] > 0
    assert result["low"] <= result["difference"] <= result["high"]
