# src/train.py
# Purpose: Train models, evaluate them, and save artifacts

import os
import json
import joblib

from datetime import datetime

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from .data_loader import load_data

from .config import (
    TARGET_COL,
    ID_COL,
    TEST_SIZE,
    RANDOM_STATE,
    MAX_ITER,
    N_ESTIMATORS,
    MAX_DEPTH,
    EVAL_METRIC
)

from .preprocessing import build_preprocessor

from .plots import (
    save_confusion_matrix,
    save_roc_curve,
    save_feature_importance
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return results, y_pred, y_proba


def train_models():

    # -------------------------
    # Load dataset
    # -------------------------
    df = load_data()

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    # -------------------------
    # Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # Preprocessing
    # -------------------------
    preprocessor = build_preprocessor()

    # -------------------------
    # Models
    # -------------------------
    models = {

        "logistic": LogisticRegression(
            max_iter=MAX_ITER
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        ),

        "xgboost": XGBClassifier(
            eval_metric=EVAL_METRIC,
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}
    results = {}

    # -------------------------
    # Train + evaluate
    # -------------------------
    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        trained_models[name] = pipeline
        results[name] = metrics

    # -------------------------
    # Select best model
    # -------------------------
    best_model_name = max(
        results,
        key=lambda x: results[x]["f1"]
    )

    best_model = trained_models[best_model_name]

    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    print(f"Best model: {best_model_name}")
    print(f"Best F1-score: {results[best_model_name]['f1']:.4f}")

    # -------------------------
    # Versioning
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(__file__)

    model_dir = os.path.join(base_dir, "..", "models")
    results_dir = os.path.join(base_dir, "..", "results")
    reports_dir = os.path.join(base_dir, "..", "reports")
    figures_dir = os.path.join(reports_dir, "figures")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------
    # Save best model
    # -------------------------
    model_path = os.path.join(
        model_dir,
        f"{best_model_name}_{timestamp}.pkl"
    )

    joblib.dump(best_model, model_path)

    # -------------------------
    # Save metrics JSON
    # -------------------------
    metrics_path = os.path.join(
        results_dir,
        f"metrics_{timestamp}.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    # -------------------------
    # Save figures
    # -------------------------
    cm_path = os.path.join(
        figures_dir,
        f"cm_{timestamp}.png"
    )

    roc_path = os.path.join(
        figures_dir,
        f"roc_{timestamp}.png"
    )

    fi_path = os.path.join(
        figures_dir,
        f"fi_top20_{timestamp}.png"
    )

    save_confusion_matrix(
        y_test,
        best_predictions,
        cm_path
    )

    save_roc_curve(
        y_test,
        best_probabilities,
        roc_path
    )

    save_feature_importance(
        best_model,
        fi_path
    )

    # -------------------------
    # Save evaluation report
    # -------------------------
    report_path = os.path.join(
        reports_dir,
        f"evaluation_{timestamp}.md"
    )

    with open(report_path, "w") as f:

        f.write("# Evaluation Report\n\n")

        f.write(f"**Run Timestamp:** {timestamp}\n\n")

        f.write("## Results\n\n")

        f.write(
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
        )

        f.write(
            "|------|----------|-----------|--------|----|---------|\n"
        )

        for model_name, metrics in results.items():

            f.write(
                f"| {model_name} "
                f"| {metrics['accuracy']:.4f} "
                f"| {metrics['precision']:.4f} "
                f"| {metrics['recall']:.4f} "
                f"| {metrics['f1']:.4f} "
                f"| {metrics['roc_auc']:.4f} |\n"
            )

        f.write("\n")

        f.write(f"## Best Model\n\n{best_model_name}\n")

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")

    return best_model, results, X_test, y_test