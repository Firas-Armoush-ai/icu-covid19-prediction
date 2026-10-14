# app/frontend/utils.py
# Helpers for the Streamlit dashboard.

# Age percentile groups used by the dataset provider (must match the API).
AGE_GROUPS = [
    "10th", "20th", "30th", "40th", "50th",
    "60th", "70th", "80th", "90th", "Above 90th",
]


def yes_no_to_int(answer):
    return 1 if answer == "Yes" else 0


def threshold_explanation(info):
    """One paragraph on how the decision threshold was chosen and how it performed."""

    test = info["test"]
    confusion = test["confusion"]
    caught = confusion["tp"]
    n_icu = test["n_icu"]

    return (
        f"Patients at or above **{info['threshold']:.0%}** are flagged. "
        f"The threshold was set on the training data so that the model would catch "
        f"{info['recall_target']:.0%} of patients who went on to need ICU. "
        f"On the held-out test set it caught {caught} of {n_icu} ICU patients "
        f"({caught / n_icu:.0%}) and flagged {test['metrics']['flagged_share']:.0%} of all patients."
    )
