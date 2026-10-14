# app/utils/inference.py
# Purpose: turn the API's user input into a full row in the training schema.

import pandas as pd

# Age percentile groups used by the dataset provider
AGE_GROUPS = [
    "10th", "20th", "30th", "40th", "50th",
    "60th", "70th", "80th", "90th", "Above 90th",
]

# In the data, AGE_ABOVE65 is 1 exactly for the 60th percentile group and above
# (352 of 353 patients; one 60th-percentile patient is 0).
FIRST_GROUP_ABOVE_65 = AGE_GROUPS.index("60th")


def age_above_65(age_group):
    return int(AGE_GROUPS.index(age_group) >= FIRST_GROUP_ABOVE_65)


def expand_lab_values(user_input, expected_columns):
    """
    Fill every summary column of a lab from its median.

    In the 0-2 h window each lab has a single measurement, so in the training
    data <LAB>_MEAN, _MIN and _MAX always equal <LAB>_MEDIAN, and <LAB>_DIFF
    and _DIFF_REL are always -1 (no change, on the provider's -1..1 scale).
    A user enters one value per lab; this reproduces how that row would look.
    """

    expected = set(expected_columns)
    expanded = dict(user_input)

    for key, value in user_input.items():
        if value is None or not key.endswith("_MEDIAN"):
            continue
        lab = key[: -len("_MEDIAN")]
        for stat in ("MEAN", "MIN", "MAX"):
            if f"{lab}_{stat}" in expected:
                expanded[f"{lab}_{stat}"] = value
        for stat in ("DIFF", "DIFF_REL"):
            if f"{lab}_{stat}" in expected:
                expanded[f"{lab}_{stat}"] = -1.0

    return expanded


def build_inference_dataframe(user_input, expected_columns):
    """
    Build a one-row dataframe in the column order the model was trained on.

    Columns the user did not provide are left empty; the model pipeline
    imputes them with values learned from the training data.
    """

    data = {col: user_input.get(col) for col in expected_columns}

    return pd.DataFrame([data], columns=list(expected_columns))
