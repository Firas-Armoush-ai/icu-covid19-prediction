"""The inference adapter turns user input into a row in the training schema."""

import json
import sys

import pandas as pd

from app.utils.inference import (
    AGE_GROUPS,
    age_above_65,
    build_inference_dataframe,
    expand_lab_values,
)
from src.config import AGE_GROUPS as TRAINING_AGE_GROUPS
from src.config import DATA_PATH, METADATA_PATH, PROJECT_ROOT

EXPECTED = ["A", "B", "C"]


def test_columns_follow_training_order():
    df = build_inference_dataframe({"C": 3, "A": 1}, EXPECTED)
    assert list(df.columns) == EXPECTED
    assert len(df) == 1


def test_missing_columns_are_empty():
    df = build_inference_dataframe({"A": 1}, EXPECTED)
    assert df.loc[0, "A"] == 1
    assert df[["B", "C"]].isna().all().all()


def test_unknown_inputs_are_dropped():
    df = build_inference_dataframe({"A": 1, "NOT_A_FEATURE": 9}, EXPECTED)
    assert "NOT_A_FEATURE" not in df.columns


def test_lab_median_fills_its_summary_columns():
    columns = ["UREA_MEDIAN", "UREA_MEAN", "UREA_MIN", "UREA_MAX", "UREA_DIFF", "PCR_MEDIAN"]
    expanded = expand_lab_values({"UREA_MEDIAN": -0.5, "PCR_MEDIAN": None}, columns)

    assert expanded["UREA_MEAN"] == expanded["UREA_MIN"] == expanded["UREA_MAX"] == -0.5
    assert expanded["UREA_DIFF"] == -1.0
    assert "UREA_DIFF_REL" not in expanded  # not a model column
    assert expanded["PCR_MEDIAN"] is None


def test_lab_expansion_matches_training_data():
    """In the training data, a lab's summary columns look exactly like the expansion."""
    df = pd.read_csv(DATA_PATH)
    columns = json.loads(METADATA_PATH.read_text())["feature_columns"]

    row = df.dropna(subset=["UREA_MEDIAN"]).iloc[0]
    expanded = expand_lab_values({"UREA_MEDIAN": row["UREA_MEDIAN"]}, columns)

    for col in ["UREA_MEAN", "UREA_MIN", "UREA_MAX", "UREA_DIFF"]:
        assert expanded[col] == row[col]


def test_age_above_65_matches_the_data():
    df = pd.read_csv(DATA_PATH)
    derived = df["AGE_PERCENTIL"].map(age_above_65)
    # One 60th-percentile patient in the data is recorded as not above 65.
    assert (derived != df["AGE_ABOVE65"]).sum() <= 1


def test_age_groups_match_the_data():
    df = pd.read_csv(DATA_PATH)
    assert sorted(df["AGE_PERCENTIL"].unique()) == sorted(AGE_GROUPS)


def import_frontend_utils(monkeypatch):
    # The Streamlit app imports its helpers as a top-level "utils" module.
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "app" / "frontend"))
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    import utils
    return utils


def test_dashboard_uses_same_age_groups(monkeypatch):
    assert import_frontend_utils(monkeypatch).AGE_GROUPS == AGE_GROUPS


def test_api_and_baseline_use_same_age_order():
    # The baseline codes age groups 0-9 in this order.
    assert TRAINING_AGE_GROUPS == AGE_GROUPS


def test_dashboard_threshold_explanation(monkeypatch):
    utils = import_frontend_utils(monkeypatch)
    info = {
        "threshold": 0.35,
        "recall_target": 0.9,
        "test": {
            "n_icu": 33,
            "confusion": {"tp": 27},
            "metrics": {"flagged_share": 0.73},
        },
    }
    text = utils.threshold_explanation(info)
    assert "35%" in text and "27 of 33" in text and "73%" in text
