"""Preprocessing is learned from the fit data only and leaves no missing values."""

import numpy as np
import pandas as pd

from src.config import DATA_PATH, ID_COL, TARGET_COL
from src.preprocessing import DropMostlyMissing, build_preprocessor


def test_drop_uses_fit_data_only():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [np.nan, np.nan, np.nan, 1]})
    test = pd.DataFrame({"a": [np.nan] * 4, "b": [1, 2, 3, 4]})

    dropper = DropMostlyMissing(threshold=0.6).fit(train)

    # "b" is 75% missing in the training data, so it is dropped even though
    # it is complete in the new data; "a" is kept even though it is empty there.
    assert list(dropper.transform(test).columns) == ["a"]


def test_threshold_is_inclusive():
    train = pd.DataFrame({"a": [np.nan, np.nan, np.nan, 1, 2], "b": [1, 2, 3, 4, 5]})  # a: 60% missing
    assert DropMostlyMissing(threshold=0.6).fit(train).columns_kept_ == ["a", "b"]


def test_pipeline_output_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])

    Xt = build_preprocessor().fit_transform(X)

    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_target_and_id_not_in_features():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])

    preprocessor = build_preprocessor().fit(X)
    names = preprocessor.get_feature_names_out()

    assert not any(TARGET_COL == n.split("__")[-1] for n in names)
    assert not any(ID_COL in n for n in names)
