# src/models.py
# Purpose: Candidate models and the full (preprocessing + model) pipeline
#
# All candidates use library defaults apart from the settings below.
# There is no hyperparameter tuning.

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from .config import (
    AGE_GROUPS,
    BASELINE_MODEL,
    DATA_PATH,
    EVAL_METRIC,
    ID_COL,
    MAX_DEPTH,
    MAX_ITER,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from .preprocessing import build_preprocessor


def make_estimator(name):

    if name == "logistic":
        return LogisticRegression(max_iter=MAX_ITER)

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )

    if name == "xgboost":
        return XGBClassifier(
            eval_metric=EVAL_METRIC,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["logistic", "random_forest", "xgboost"]


def build_pipeline(name):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", make_estimator(name)),
    ])


# Each model is evaluated as it is and with sigmoid (Platt) calibration.
CALIBRATION_SUFFIX = "+sigmoid"
CANDIDATES = MODEL_NAMES + [name + CALIBRATION_SUFFIX for name in MODEL_NAMES]


def build_baseline():
    """Logistic regression on the age group alone, coded 0 (10th) to 9 (above 90th)."""

    return Pipeline([
        ("age", ColumnTransformer([
            ("age_group", OrdinalEncoder(categories=[AGE_GROUPS]), ["AGE_PERCENTIL"]),
        ])),
        ("model", LogisticRegression(max_iter=MAX_ITER)),
    ])


def build_model(candidate):
    """
    Build a candidate such as "random_forest" or "random_forest+sigmoid".

    Calibrated candidates wrap the whole pipeline in CalibratedClassifierCV:
    within the data it is fitted on, 5 stratified folds each fit the pipeline
    on 4/5 and learn a sigmoid mapping on the remaining 1/5. Predictions
    average the 5 calibrated pipelines.
    """

    if candidate == BASELINE_MODEL:
        return build_baseline()

    if candidate.endswith(CALIBRATION_SUFFIX):
        base = candidate[: -len(CALIBRATION_SUFFIX)]
        return CalibratedClassifierCV(
            build_pipeline(base),
            method="sigmoid",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )

    return build_pipeline(candidate)


def load_split():
    """
    Load the modelling data and make the train/test split.

    The split is fixed (stratified, seed 42). The test set is only used once,
    for the final evaluation; everything else uses the training set.
    """

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
