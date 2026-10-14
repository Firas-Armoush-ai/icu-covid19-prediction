# src/preprocessing.py
# Purpose: Build the preprocessing pipeline
#
# Every step is learned from the data passed to fit(), so in cross-validation
# each fold's preprocessing sees only that fold's training rows.

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import CATEGORICAL_COLS, MISSING_THRESHOLD


class DropMostlyMissing(BaseEstimator, TransformerMixin):
    """Drop columns whose share of missing values in the fit data exceeds a threshold."""

    def __init__(self, threshold=MISSING_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        missing_ratio = X.isna().mean()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columns_kept_ = missing_ratio.index[missing_ratio <= self.threshold].tolist()
        self.columns_dropped_ = missing_ratio.index[missing_ratio > self.threshold].tolist()

        return self

    def transform(self, X):
        check_is_fitted(self, "columns_kept_")
        return X[self.columns_kept_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_kept_")
        return np.asarray(self.columns_kept_, dtype=object)


# Column selectors are module-level functions so the pipeline can be pickled.
def categorical_columns(X):
    return [col for col in CATEGORICAL_COLS if col in X.columns]


def numeric_columns(X):
    return [col for col in X.columns if col not in CATEGORICAL_COLS]


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    return Pipeline([
        ("drop_missing", DropMostlyMissing()),
        ("columns", column_transformer)
    ])
