# src/threshold.py
# Purpose: Choose a decision threshold from out-of-fold predictions so that
#          it meets a recall target, and report what it costs in precision.

import numpy as np
import pandas as pd


def threshold_for_recall(y_true, proba, target):
    """
    Highest threshold whose recall is at least `target`.

    Using the highest such threshold keeps as much precision as possible
    while still meeting the recall target.
    """

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    candidates = np.unique(proba)[::-1]  # high to low
    n_pos = y_true.sum()

    for t in candidates:
        if (y_true[proba >= t] == 1).sum() / n_pos >= target:
            return float(t)

    return float(candidates[-1])


def operating_point(y_true, proba, threshold):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(proba) >= threshold

    tp = int((y_pred & (y_true == 1)).sum())
    fp = int((y_pred & (y_true == 0)).sum())
    fn = int((~y_pred & (y_true == 1)).sum())
    tn = int((~y_pred & (y_true == 0)).sum())

    return {
        "threshold": float(threshold),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "specificity": tn / (tn + fp),
        "flagged_share": (tp + fp) / len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def threshold_table(y_true, oof, targets):
    """
    For each recall target, choose a threshold separately in every CV repeat
    (each repeat gives one out-of-fold probability per patient) and summarise
    the operating points across repeats.

    The recommended threshold for a target is the median of the per-repeat
    thresholds.
    """

    rows = []

    for target in targets:
        points = []
        for r in range(oof.shape[1]):
            t = threshold_for_recall(y_true, oof[:, r], target)
            points.append(operating_point(y_true, oof[:, r], t))

        points = pd.DataFrame(points)
        rows.append({
            "recall_target": target,
            "threshold": float(points["threshold"].median()),
            **{f"{m}_mean": float(points[m].mean()) for m in ["recall", "precision", "specificity", "flagged_share"]},
            **{f"{m}_std": float(points[m].std(ddof=1)) for m in ["precision", "specificity"]},
        })

    return pd.DataFrame(rows)
