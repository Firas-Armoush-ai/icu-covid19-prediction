# src/evaluate.py
# Purpose: Evaluate the final model on the held-out test set, with bootstrap
#          confidence intervals.

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .threshold import operating_point

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

TEST_METRICS = ["roc_auc", "pr_auc", "brier", "recall", "precision", "specificity", "f1"]


def test_metrics(y_true, proba, threshold):

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    point = operating_point(y_true, proba, threshold)
    precision, recall = point["precision"], point["recall"]

    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "recall": recall,
        "precision": precision,
        "specificity": point["specificity"],
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flagged_share": point["flagged_share"],
        "tp": point["tp"], "fp": point["fp"], "fn": point["fn"], "tn": point["tn"],
    }


def bootstrap_ci(y_true, proba, threshold, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    95% percentile intervals from resampling test patients with replacement.

    The model and threshold stay fixed; only the test sample is resampled, so
    the intervals show how much the estimates depend on which patients
    happened to land in a test set of this size.
    """

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)

    samples = {m: [] for m in TEST_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        if y_b.min() == y_b.max():  # AUC needs both classes
            continue
        metrics = test_metrics(y_b, proba[idx], threshold)
        for m in TEST_METRICS:
            samples[m].append(metrics[m])

    return {
        m: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for m, values in samples.items()
    }


def evaluate_on_test(y_true, proba, threshold):

    point = test_metrics(y_true, proba, threshold)
    ci = bootstrap_ci(y_true, proba, threshold)

    return {
        "n_patients": int(len(y_true)),
        "n_icu": int(np.sum(y_true)),
        "threshold": float(threshold),
        "metrics": {m: float(point[m]) for m in TEST_METRICS + ["flagged_share"]},
        "confusion": {k: int(point[k]) for k in ["tp", "fp", "fn", "tn"]},
        "ci95": ci,
        "n_bootstrap": N_BOOTSTRAP,
    }
