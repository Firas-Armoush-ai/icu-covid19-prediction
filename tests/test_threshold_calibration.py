"""Threshold selection and calibration helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.calibration import calibration_slope_intercept, calibration_summary
from src.config import DATA_PATH, ID_COL, TARGET_COL
from src.models import CANDIDATES, build_model
from src.threshold import operating_point, threshold_for_recall, threshold_table

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1])


def test_threshold_meets_recall_target():
    t = threshold_for_recall(Y, P, 0.75)
    assert t == 0.6  # 0.9, 0.8, 0.6 -> 3 of 4 positives
    assert operating_point(Y, P, t)["recall"] >= 0.75


def test_threshold_is_highest_that_meets_target():
    # Full recall needs the lowest positive score (0.3) as the threshold.
    assert threshold_for_recall(Y, P, 1.0) == 0.3


def test_operating_point_counts():
    point = operating_point(Y, P, 0.6)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (3, 1, 1, 3)
    assert point["precision"] == 0.75
    assert point["flagged_share"] == 0.5


def test_threshold_table_uses_every_repeat():
    oof = np.column_stack([P, P])
    table = threshold_table(Y, oof, [0.75, 1.0])
    assert list(table["threshold"]) == [0.6, 0.3]


def test_slope_near_one_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = rng.binomial(1, p)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_slope_below_one_for_overconfident_predictions():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.3, 0.7, 20_000)
    y = rng.binomial(1, true_p)
    overconfident = np.where(true_p > 0.5, 0.95, 0.05)
    slope, _ = calibration_slope_intercept(y, overconfident)
    assert slope < 0.5


def test_calibration_summary_keys():
    summary = calibration_summary(Y, np.column_stack([P, P]))
    assert {"brier_mean", "slope_mean", "brier_reference"} <= set(summary)


def test_calibrated_candidate_fits_and_predicts():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]

    model = build_model("random_forest+sigmoid")
    assert isinstance(model, CalibratedClassifierCV)

    proba = model.fit(X, y).predict_proba(X.head(5))[:, 1]
    assert ((proba > 0) & (proba < 1)).all()


def test_every_candidate_builds():
    for name in CANDIDATES:
        build_model(name)
