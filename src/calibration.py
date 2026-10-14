# src/calibration.py
# Purpose: Check whether predicted probabilities match observed ICU rates,
#          using out-of-fold predictions from cross-validation.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_slope_intercept(y_true, proba, eps=1e-6):
    """
    Logistic recalibration: regress the outcome on logit(p).

    A well-calibrated model has slope close to 1 and intercept close to 0.
    Slope < 1 means predictions are too extreme (overconfident);
    slope > 1 means they are too close to the middle.
    """

    p = np.clip(np.asarray(proba), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    fit = LogisticRegression(penalty=None).fit(logit, y_true)

    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def calibration_summary(y_true, oof):
    """Brier score and calibration slope/intercept, per CV repeat, summarised."""

    y_true = np.asarray(y_true)
    prevalence = y_true.mean()

    briers, slopes, intercepts, mean_preds = [], [], [], []

    for r in range(oof.shape[1]):
        p = oof[:, r]
        briers.append(brier_score_loss(y_true, p))
        slope, intercept = calibration_slope_intercept(y_true, p)
        slopes.append(slope)
        intercepts.append(intercept)
        mean_preds.append(p.mean())

    return {
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers, ddof=1)),
        # Brier score of always predicting the training prevalence
        "brier_reference": float(prevalence * (1 - prevalence)),
        "slope_mean": float(np.mean(slopes)),
        "slope_std": float(np.std(slopes, ddof=1)),
        "intercept_mean": float(np.mean(intercepts)),
        "mean_predicted": float(np.mean(mean_preds)),
        "observed_rate": float(prevalence),
    }


def save_reliability_plot(y_true, oof_by_model, save_path, title, n_bins=8):
    """Reliability curve per model, pooling all CV repeats."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")

    y_true = np.asarray(y_true)

    for name, oof in oof_by_model.items():
        y_rep = np.tile(y_true, oof.shape[1])
        p_rep = oof.T.ravel()
        frac_pos, mean_pred = calibration_curve(y_rep, p_rep, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed ICU rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
