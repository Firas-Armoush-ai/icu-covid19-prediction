# src/plots.py
# Purpose: Generate and save evaluation figures

import matplotlib

matplotlib.use("Agg")

import re

import matplotlib.pyplot as plt
import pandas as pd

from .config import CATEGORICAL_COLS

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def save_confusion_matrix(y_test, y_pred, save_path, title="Confusion Matrix"):

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No ICU", "ICU"]
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_roc_curve(y_test, y_proba, save_path, title="ROC Curve"):

    RocCurveDisplay.from_predictions(y_test, y_proba)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def fitted_pipelines(model):
    """The fitted (preprocessing + model) pipelines inside a model.

    A calibrated model holds one pipeline per internal fold; a plain pipeline
    is returned as a single-item list.
    """

    if hasattr(model, "calibrated_classifiers_"):
        return [c.estimator for c in model.calibrated_classifiers_]
    return [model]


# Each lab and vital sign comes as several summary columns. In the 0-2 h window
# most patients have one measurement, so MEAN, MEDIAN, MIN and MAX are identical
# and DIFF is constant; grouping them avoids splitting one measurement's
# importance across copies.
STAT_SUFFIX = re.compile(r"_(MEDIAN|MEAN|MIN|MAX|DIFF_REL|DIFF)$")


def source_measurement(encoded_name):
    """Map an encoded feature name back to the measurement it comes from.

    "num__UREA_MEDIAN" -> "UREA"; "cat__AGE_PERCENTIL_10th" -> "AGE_PERCENTIL".
    """

    name = encoded_name.split("__", 1)[-1]
    for col in CATEGORICAL_COLS:
        if name.startswith(col + "_"):
            return col
    return STAT_SUFFIX.sub("", name)


def feature_importance(model):
    """Impurity-based importances, averaged over the fitted pipelines and
    summed per measurement (one-hot columns and summary statistics combined).

    Returns None if the underlying estimator has no feature_importances_.
    """

    per_pipeline = []

    for pipeline in fitted_pipelines(model):
        estimator = pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            return None
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        per_pipeline.append(pd.Series(estimator.feature_importances_, index=names))

    # A feature missing from one pipeline (dropped there for missingness) counts as 0.
    encoded = pd.concat(per_pipeline, axis=1).fillna(0).mean(axis=1)

    return (
        encoded.groupby(source_measurement)
        .sum()
        .sort_values(ascending=False)
    )


def save_feature_importance(model, save_path, top_n=20):

    importance = feature_importance(model)

    if importance is None:
        return None

    top = importance.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.title(f"Top {top_n} measurements by importance (impurity-based, summed per measurement)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return top
