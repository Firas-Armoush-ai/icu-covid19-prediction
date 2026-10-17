# src/config.py
# Purpose: Central place for constants

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_COL = "ICU"
ID_COL = "PATIENT_VISIT_IDENTIFIER"
WINDOW_COL = "WINDOW"

# Only the first observation window is used for prediction
EARLY_WINDOW = "0-2"

CATEGORICAL_COLS = [
    'AGE_PERCENTIL',
    'AGE_ABOVE65',
    'GENDER',
    'DISEASE GROUPING 1',
    'DISEASE GROUPING 2',
    'DISEASE GROUPING 3',
    'DISEASE GROUPING 4',
    'DISEASE GROUPING 5',
    'DISEASE GROUPING 6',
    'HTN',
    'IMMUNOCOMPROMISED',
    'OTHER'
]

# Age percentile groups used by the data provider, youngest first
AGE_GROUPS = [
    "10th", "20th", "30th", "40th", "50th",
    "60th", "70th", "80th", "90th", "Above 90th",
]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Kaggle_Sirio_Libanes_ICU_Prediction.xlsx"
# SHA-256 of the original file as downloaded from Kaggle (version 4); tests check it
RAW_DATA_SHA256 = "325889ae11b45a156ba71afad498f0eff5726c9abd1f411afca1865aafd8ebd1"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "df_model.csv"

# Columns with a larger share of missing values (in the training data) are dropped
MISSING_THRESHOLD = 0.6

# Logistic Regression hyperparameters
MAX_ITER = 1000

# RandomForest hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS= 100
MAX_DEPTH= None

# XGBoost hyperparameters
EVAL_METRIC = "logloss"

# Test size
TEST_SIZE = 0.2

# Final model and decision threshold, chosen from the cross-validation report
# (reports/cv_report.md). Set after reviewing those results.
FINAL_MODEL = "random_forest+sigmoid"
RECALL_TARGET = 0.90

# One-feature baseline: logistic regression on the age group alone.
# Reported next to the candidates; a model that cannot beat it has learned
# nothing beyond age.
BASELINE_MODEL = "age_only"

# Saved model and its metadata (threshold, columns, evaluation results)
MODEL_PATH = PROJECT_ROOT / "models" / "icu_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
