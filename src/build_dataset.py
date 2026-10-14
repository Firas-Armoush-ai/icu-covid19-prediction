# src/build_dataset.py
# Purpose: Build the patient-level modelling dataset from the raw file
#
# Run from the project root:
#     python -m src.build_dataset

import pandas as pd

from .config import (
    DATA_PATH,
    EARLY_WINDOW,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
    WINDOW_COL,
)


def build_model_dataset(raw):
    """
    Return one row per patient, using only the first observation window.

    * Features: the patient's measurements from the 0-2 h window.
    * Target: 1 if the patient was admitted to ICU in any window.
    * Patients whose ICU flag is already 1 in the 0-2 h window are excluded.
      The data provider warns not to use rows where the target is present,
      because it is unknown whether those measurements were taken before
      or after ICU admission.
    """

    ever_icu = raw.groupby(ID_COL)[TARGET_COL].max()

    early = raw[raw[WINDOW_COL] == EARLY_WINDOW]
    early = early[early[TARGET_COL] == 0]

    if early[ID_COL].duplicated().any():
        raise ValueError("Expected one 0-2 h row per patient")

    df = early.drop(columns=[WINDOW_COL, TARGET_COL]).copy()
    df[TARGET_COL] = df[ID_COL].map(ever_icu).astype(int)

    return df.reset_index(drop=True)


def main():

    raw = pd.read_excel(RAW_DATA_PATH)
    df = build_model_dataset(raw)

    n_patients = raw[ID_COL].nunique()
    n_excluded = n_patients - len(df)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Raw file:          {raw.shape[0]} rows, {n_patients} patients")
    print(f"Excluded:          {n_excluded} patients already in ICU at 0-2 h")
    print(f"Modelling dataset: {df.shape[0]} patients, {df.shape[1] - 2} features")
    print(f"Target:            {df[TARGET_COL].value_counts().sort_index().to_dict()}")
    print(f"Saved to:          {DATA_PATH}")


if __name__ == "__main__":
    main()
