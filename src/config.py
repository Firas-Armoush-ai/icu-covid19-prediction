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

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Kaggle_Sirio_Libanes_ICU_Prediction.xlsx"
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
