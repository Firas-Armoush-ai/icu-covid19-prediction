# src/plots.py
# Purpose: Generate and save evaluation figures

import matplotlib

matplotlib.use("Agg")

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from .config import CATEGORICAL_COLS


# --------------------------------------------------
# Shared style for the figures shown in the README and the dashboard
# --------------------------------------------------
# Categorical colours: the first three slots of a palette checked for
# colour-blind separation; everything else is neutral grey.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#8f8d87", "#d4d2cc"
TEXT, TEXT_2 = "#0b0b0b", "#52514e"
SURFACE = "#ffffff"
# One-hue ramp (light to dark blue) for counts
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])

DPI = 150

DISPLAY_NAMES = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "age_only": "Age group only (baseline)",
}


def display_name(candidate, suffix="+sigmoid"):
    if candidate.endswith(suffix):
        return DISPLAY_NAMES[candidate[: -len(suffix)]] + " + calibration"
    return DISPLAY_NAMES[candidate]


def _figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY_LIGHT)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    return fig, ax


def _titles(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9, color=TEXT_2)


def _save(fig, save_path):
    fig.savefig(save_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def save_confusion_matrix(y_test, y_pred, save_path, threshold):
    """Test-set confusion matrix, ICU patients on the top row."""

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    cells = np.array([[tp, fn], [fp, tn]])
    words = np.array([["caught", "missed"], ["false alarm", "correctly cleared"]])

    fig, ax = _figure(6.4, 4.6)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.08)
    ax.imshow(cells, cmap=BLUES, vmin=0, vmax=cells.max())

    for i in range(2):
        for j in range(2):
            dark = cells[i, j] > 0.6 * cells.max()
            colour = SURFACE if dark else TEXT
            ax.text(j, i - 0.08, str(cells[i, j]), ha="center", va="center", fontsize=22,
                    fontweight="bold", color=colour)
            ax.text(j, i + 0.22, words[i, j], ha="center", va="center", fontsize=9, color=colour)

    ax.set_xticks([0, 1], ["Flagged for review", "Not flagged"])
    ax.set_yticks([0, 1], [f"Admitted to ICU ({tp + fn})", f"Not admitted ({fp + tn})"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=10, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n = cells.sum()
    _titles(
        fig,
        f"Test set: {n} patients at threshold {threshold:.3f}",
        f"Caught {tp} of {tp + fn} ICU patients ({tp / (tp + fn):.0%}); "
        f"flagged {tp + fp} of {n} ({(tp + fp) / n:.0%}).",
    )
    _save(fig, save_path)


def save_model_comparison(fold_results, final_model, baseline, save_path, seed=0):
    """
    ROC-AUC of every candidate across all CV folds: one dot per fold, with the
    mean and one standard deviation. The final model is blue, the age-only
    baseline orange, the rest grey.
    """

    order = (
        fold_results.groupby("model")["roc_auc"].mean().sort_values().index.tolist()
    )
    rng = np.random.default_rng(seed)

    fig, ax = _figure(8.4, 4.6)
    fig.subplots_adjust(left=0.33, right=0.9, top=0.8, bottom=0.12)

    for row, name in enumerate(order):
        values = fold_results.loc[fold_results["model"] == name, "roc_auc"].to_numpy()
        colour = BLUE if name == final_model else ORANGE if name == baseline else GREY
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(values, row + jitter, s=12, color=colour, alpha=0.35, linewidths=0)
        mean, sd = values.mean(), values.std(ddof=1)
        ax.plot([mean - sd, mean + sd], [row, row], color=colour, lw=2.2, solid_capstyle="round")
        ax.scatter([mean], [row], s=70, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(1.005, row, f"{mean:.3f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color=TEXT, fontweight="bold" if name in (final_model, baseline) else None)

    labels = [display_name(n) + (" (final)" if n == final_model else "") for n in order]
    ax.set_yticks(range(len(order)), labels)
    for tick, name in zip(ax.get_yticklabels(), order):
        tick.set_color(TEXT)
        if name in (final_model, baseline):
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, labelsize=9.5)

    ax.axvline(0.5, color=GREY_LIGHT, lw=1, ls=(0, (2, 2)), zorder=0)
    ax.text(0.505, len(order) - 0.45, "chance", fontsize=8, color=TEXT_2, va="center")
    ax.set_xlim(0.38, 0.9)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlabel("ROC-AUC on the validation fold", fontsize=9, color=TEXT_2)
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(1.005, len(order) - 0.45, "mean", transform=ax.get_yaxis_transform(), fontsize=8,
            color=TEXT_2, va="center")

    n_folds = fold_results.groupby("model").size().iloc[0]
    _titles(
        fig,
        "The models overlap, and age alone does as well",
        f"Cross-validated ROC-AUC on the training split: {n_folds} folds per model. "
        "Dots are folds; the bar is the mean \u00b1 1 SD.",
    )
    _save(fig, save_path)


def save_threshold_tradeoff(table, chosen_target, flag_all_precision, save_path):
    """
    Precision, specificity and share of patients flagged at each recall target,
    from out-of-fold predictions. `table` is the output of threshold_table().
    """

    fig, ax = _figure(8.4, 4.8)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.8, bottom=0.12)

    x = table["recall_target"]
    series = [
        ("flagged_share_mean", "Patients flagged", AQUA),
        ("precision_mean", "Precision", BLUE),
        ("specificity_mean", "Specificity", ORANGE),
    ]
    for column, label, colour in series:
        ax.plot(x, table[column], color=colour, lw=2, label=label)
        ax.text(x.iloc[-1] + 0.004, table[column].iloc[-1], label, color=TEXT, fontsize=9, va="center")

    ax.axhline(flag_all_precision, color=GREY, lw=1, ls=(0, (2, 2)))
    ax.text(chosen_target + 0.006, flag_all_precision - 0.02,
            f"flag everyone: precision {flag_all_precision:.0%}", fontsize=8, color=TEXT_2, va="top")

    chosen = table.loc[(table["recall_target"] - chosen_target).abs().idxmin()]
    ax.axvline(chosen_target, color=TEXT_2, lw=1)
    for column, _, colour in series:
        ax.scatter([chosen_target], [chosen[column]], s=60, color=colour, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
    ax.annotate(
        f"Chosen: {chosen_target:.0%} recall\n"
        f"threshold {chosen['threshold']:.3f}\n"
        f"flags {chosen['flagged_share_mean']:.0%}, precision {chosen['precision_mean']:.0%},\n"
        f"specificity {chosen['specificity_mean']:.0%}",
        xy=(chosen_target, 0.97), xytext=(chosen_target - 0.012, 0.97), ha="right", va="top",
        fontsize=8.5, color=TEXT,
    )

    ax.set_xlim(x.iloc[0], x.iloc[-1])
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Recall target: share of ICU patients the threshold must catch", fontsize=9, color=TEXT_2)
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    _titles(
        fig,
        "What each recall target costs",
        "Final model, out-of-fold predictions on the training split; mean over 10 CV repeats.",
    )
    _save(fig, save_path)


def save_calibration_before_after(y_true, oof_before, oof_after, fits, save_path, n_bins=8):
    """
    Calibration of the final model's base estimator before and after sigmoid
    calibration, from out-of-fold predictions pooled over the CV repeats.

    Lines: the logistic recalibration curve, observed = sigmoid(a + b * logit(p)),
    using the mean slope b and intercept a from the calibration summary (`fits`:
    the before and after summaries), drawn over the central 95% of predictions.
    Dots: observed ICU rate in equal-size bins of predicted probability.
    """

    y_true = np.asarray(y_true)

    fig, ax = _figure(6.8, 5.6)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.82, bottom=0.11)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (2, 2)))
    ax.text(0.83, 0.79, "perfect calibration", rotation=40, fontsize=8, color=TEXT_2,
            rotation_mode="anchor", ha="center")

    curves = [
        (oof_before, fits[0], "Before (random forest)", ORANGE),
        (oof_after, fits[1], "After (+ sigmoid calibration)", BLUE),
    ]
    for oof, fit, label, colour in curves:
        p_rep = oof.T.ravel()
        y_rep = np.tile(y_true, oof.shape[1])
        observed, predicted = calibration_curve(y_rep, p_rep, n_bins=n_bins, strategy="quantile")
        ax.scatter(predicted, observed, s=30, color=colour, alpha=0.45, linewidths=0)

        lo, hi = np.percentile(p_rep, [2.5, 97.5])
        grid = np.linspace(lo, hi, 200)
        logit = np.log(grid / (1 - grid))
        curve = 1 / (1 + np.exp(-(fit["intercept_mean"] + fit["slope_mean"] * logit)))
        ax.plot(grid, curve, color=colour, lw=2.4,
                label=f"{label}, slope {fit['slope_mean']:.2f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Predicted probability of ICU", fontsize=9, color=TEXT_2)
    ax.set_ylabel("Observed ICU rate", fontsize=9, color=TEXT_2)
    ax.grid(color=GREY_LIGHT, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)

    _titles(
        fig,
        "Calibration fixes overconfident probabilities",
        "Out-of-fold predictions, training split, 10 repeats. Lines: fitted calibration curve over\n"
        f"the central 95% of predictions. Dots: observed rate in {n_bins} equal-size bins. Ideal slope: 1.",
    )
    _save(fig, save_path)


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
