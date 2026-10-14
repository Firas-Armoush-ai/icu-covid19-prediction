# run.py
# Purpose: Train the final model, evaluate it on the test set and save it.
#
# Model comparison happens first, in: python -m src.cross_validate

from src.config import METADATA_PATH, MODEL_PATH, PROJECT_ROOT
from src.train import train_final


def main():

    _, metadata = train_final()

    test = metadata["test"]
    m, ci = test["metrics"], test["ci95"]

    print(f"Model:      {metadata['model']}")
    print(f"Threshold:  {metadata['threshold']:.3f} (recall target {metadata['recall_target']:.2f})")
    print(f"Test set:   {test['n_patients']} patients, {test['n_icu']} admitted to ICU")
    print()
    for metric in ["roc_auc", "pr_auc", "brier", "recall", "precision", "specificity", "f1"]:
        print(f"  {metric:<12} {m[metric]:.3f}  (95% CI {ci[metric]['low']:.3f}-{ci[metric]['high']:.3f})")
    print()
    print(f"Model saved to:    {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metadata saved to: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to:   reports/evaluation_test.md")


if __name__ == "__main__":
    main()
