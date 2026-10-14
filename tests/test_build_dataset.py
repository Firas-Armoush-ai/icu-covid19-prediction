"""The modelling dataset uses only the 0-2 h window, one row per patient."""

import hashlib

import pandas as pd
import pytest

from src.build_dataset import build_model_dataset
from src.config import (
    DATA_PATH,
    EARLY_WINDOW,
    ID_COL,
    RAW_DATA_PATH,
    RAW_DATA_SHA256,
    TARGET_COL,
    WINDOW_COL,
)

WINDOWS = ["0-2", "2-4", "4-6", "6-12", "ABOVE_12"]


def make_raw(icu_by_patient):
    """Synthetic raw data: one row per patient per window.

    LAB is set to the window index (0 for 0-2 h, 1 for 2-4 h, ...) so the
    test can tell which window a value came from.
    """
    rows = []
    for patient, icu_flags in icu_by_patient.items():
        for i, (window, icu) in enumerate(zip(WINDOWS, icu_flags)):
            rows.append({ID_COL: patient, "LAB": float(i), WINDOW_COL: window, TARGET_COL: icu})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic():
    raw = make_raw({
        1: [0, 0, 0, 0, 0],  # never in ICU
        2: [0, 0, 0, 1, 1],  # admitted later -> positive
        3: [1, 1, 1, 1, 1],  # already in ICU at 0-2 h -> excluded
        4: [0, 1, 1, 1, 1],  # admitted at 2-4 h -> positive
    })
    return build_model_dataset(raw)


def test_one_row_per_patient(synthetic):
    assert synthetic[ID_COL].is_unique


def test_only_early_window_values(synthetic):
    assert (synthetic["LAB"] == WINDOWS.index(EARLY_WINDOW)).all()


def test_patients_already_in_icu_are_excluded(synthetic):
    assert set(synthetic[ID_COL]) == {1, 2, 4}


def test_target_is_admission_in_any_window(synthetic):
    target = synthetic.set_index(ID_COL)[TARGET_COL].to_dict()
    assert target == {1: 0, 2: 1, 4: 1}


def test_window_column_removed(synthetic):
    assert WINDOW_COL not in synthetic.columns


@pytest.fixture(scope="module")
def real():
    return build_model_dataset(pd.read_excel(RAW_DATA_PATH))


def test_real_dataset_shape(real):
    assert len(real) == 353
    assert real[TARGET_COL].sum() == 163
    assert real[ID_COL].is_unique


def test_committed_csv_matches_rebuild(real):
    committed = pd.read_csv(DATA_PATH)
    pd.testing.assert_frame_equal(committed, real, check_dtype=False)


def test_raw_file_is_the_kaggle_original():
    """Guards against accidental edits (e.g. opening and re-saving the file in Excel)."""
    digest = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    assert digest == RAW_DATA_SHA256
