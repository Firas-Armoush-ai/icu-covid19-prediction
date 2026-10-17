# ICU Prediction for COVID-19 Patients (Early-Stage Machine Learning System)

[![tests](https://github.com/Firas-Armoush-ai/icu-covid19-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/Firas-Armoush-ai/icu-covid19-prediction/actions/workflows/tests.yml)

🚀 **Live Demo:**
https://icu-covid19-prediction.onrender.com/

📘 **API documentation:** the live site serves the Streamlit dashboard only. The FastAPI backend runs privately inside the same container, so its Swagger docs are available when you run the app locally, at http://127.0.0.1:8000/docs (see [Run Application Locally](#run-application-locally)).

---

## Overview

This project develops an end-to-end machine learning system to predict whether a patient with confirmed COVID-19 will require Intensive Care Unit (ICU) admission using only early-stage clinical data collected within the first 0–2 hours.

The project evolved from exploratory notebook experimentation into a modular, reproducible, deployment-ready ML application featuring:

* Leakage-safe, patient-level dataset built in code
* Repeated cross-validation, probability calibration and a recall-targeted decision threshold
* A single, untouched test set with bootstrap confidence intervals
* FastAPI inference backend and Streamlit dashboard
* Tests and CI (GitHub Actions)
* Dockerized deployment on Render

The objective is to support early risk identification and clinically realistic ICU prediction workflows.

---

## Business / Clinical Problem

Early identification of patients at risk of severe deterioration is critical in COVID-19 management.

Accurate ICU risk prediction can support:

* Prioritized patient monitoring
* Efficient ICU resource allocation
* Earlier clinical intervention
* Improved operational planning under constrained resources

This project focuses on clinically realistic early prediction, following the principle:

> “The earlier, the better.”

---

## Dataset

### Dataset source

| | |
| --- | --- |
| **Name** | COVID-19 – Clinical Data to assess diagnosis |
| **Provider** | Hospital Sírio-Libanês (Data Intelligence Team), São Paulo and Brasília, Brazil |
| **Where** | [Kaggle](https://www.kaggle.com/datasets/S%C3%ADrio-Libanes/covid19), version 4, file `Kaggle_Sirio_Libanes_ICU_Prediction.xlsx` (1,925 rows × 231 columns) |
| **Period covered** | 28 January – 16 April 2020 |
| **Licence** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use, attribution required) |

The provider publishes no DOI or formal citation. Suggested citation:

> Hospital Sírio-Libanês Data Intelligence Team (2020). *COVID-19 – Clinical Data to assess diagnosis* (version 4) [Data set]. Kaggle. https://www.kaggle.com/datasets/S%C3%ADrio-Libanes/covid19. Licensed under CC BY-NC 4.0.

The provider states that the data were anonymised and that every numeric column was min-max scaled to the range −1 to 1 before publication. See [Licence](#licence) for how the data in this repository may be reused.

### Dataset contents

The dataset contains clinical and laboratory measurements for **385 confirmed COVID-19 patients**, each observed across multiple temporal windows:

* 0–2 hours
* 2–4 hours
* 4–6 hours
* 6–12 hours
* Above 12 hours

### Final Modeling Dataset

`src/build_dataset.py` builds `data/processed/df_model.csv` from the raw file:

* Only the first observation window (0–2 hours) is used, one row per patient
* The target is 1 if the patient was admitted to ICU in **any** window
* **32 patients whose ICU flag is already set in the 0–2 hour window are excluded.** The data provider warns not to use rows where the target is present, because it is unknown whether those measurements were taken before or after ICU admission. Keeping them would leak the outcome into the features.

### Final Dataset Characteristics

* 353 patients (385 minus the 32 excluded): 163 admitted to ICU, 190 not
* 228 candidate input features. Columns with more than 60% missing values are dropped inside the model pipeline, using the training data only. On the full dataset this keeps 192 features (212 after one-hot encoding); all vital-sign columns are dropped because about 70% of patients have no vital signs recorded in the first two hours.

> `df_model.csv` shape is `(353, 230)`: 228 features, 1 target column and 1 patient identifier column.

---

## Methodology

### Data Strategy

The project applies a clinically realistic early prediction strategy:

* Restricted analysis to the first observation window (0–2 hours)
* Constructed a patient-level ICU target
* Excluded patients already in ICU during that window, whose measurements may come from after admission
* One row per patient, so the same patient cannot appear in both training and test data

### Preprocessing Pipeline

The preprocessing workflow was implemented using reusable Scikit-learn pipelines.

#### Numerical Features

* Median imputation
* Standard scaling

#### Categorical Features

* Most frequent imputation
* One-hot encoding

### Feature Engineering

* Removed features with >60% missing values, as a pipeline step fitted on the training data only
* Structured feature groups explicitly
* Built reusable preprocessing architecture with:

  * `Pipeline`
  * `ColumnTransformer`

---

## Modeling and Evaluation

### Candidate models

* Logistic Regression (baseline)
* Random Forest
* XGBoost

Each model uses library defaults (no hyperparameter tuning) and is evaluated both as it is and wrapped in **sigmoid (Platt) calibration**.

### Evaluation design

1. **Fixed hold-out split.** 80/20 stratified split: 282 training patients (130 ICU), 71 test patients (33 ICU). The test set is not used until the very end.
2. **Model comparison on the training split only.** Repeated stratified 5-fold cross-validation, 10 repeats (50 folds). Every fold refits the whole pipeline, including imputation, scaling, encoding, the missing-column filter and calibration, so nothing is learned from validation rows. Script: `python -m src.cross_validate`; full report: [`reports/cv_report.md`](reports/cv_report.md).
3. **Decision threshold from a recall target.** Missed ICU cases matter most, so the threshold is the highest one that reaches **90% recall** on out-of-fold predictions (chosen per CV repeat; median used): **0.352**.
4. **Calibration check.** Brier score, calibration slope and reliability curves on out-of-fold predictions.
5. **One test-set evaluation.** The final model and threshold were committed before the test set was used, then evaluated once, with 95% bootstrap intervals. Script: `python run.py`; report: [`reports/evaluation_test.md`](reports/evaluation_test.md).

---

## Results

### Cross-validation (training split, 50 folds, mean ± SD across folds)

| Model | ROC-AUC | PR-AUC | Brier score | Calibration slope |
| --- | --- | --- | --- | --- |
| logistic | 0.680 ± 0.079 | 0.666 ± 0.084 | 0.248 ± 0.008 | 0.14 ± 0.03 |
| random_forest | 0.686 ± 0.063 | 0.619 ± 0.073 | 0.235 ± 0.009 | 0.24 ± 0.07 |
| xgboost | 0.695 ± 0.070 | 0.651 ± 0.077 | 0.252 ± 0.012 | 0.31 ± 0.03 |
| logistic+sigmoid | 0.667 ± 0.083 | 0.653 ± 0.086 | 0.236 ± 0.003 | 1.14 ± 0.32 |
| **random_forest+sigmoid** (final) | 0.695 ± 0.064 | 0.627 ± 0.073 | 0.225 ± 0.004 | 1.17 ± 0.22 |
| xgboost+sigmoid | 0.709 ± 0.067 | 0.661 ± 0.078 | 0.222 ± 0.004 | 1.32 ± 0.25 |

* **Discrimination is similar across models.** The differences in ROC-AUC are smaller than the fold-to-fold spread.
* **Uncalibrated models are badly overconfident.** Calibration slopes of 0.14–0.31 (ideal: 1) and Brier scores no better than always predicting the ICU rate (0.248). Sigmoid calibration fixes this.

What the 90% recall target costs (final model, out-of-fold):

| Recall target | Threshold | Precision | Specificity | Patients flagged |
| --- | --- | --- | --- | --- |
| 0.80 | 0.412 | 0.583 | 0.509 | 63% |
| 0.85 | 0.377 | 0.554 | 0.412 | 71% |
| **0.90** | **0.352** | **0.539** | **0.340** | **77%** |
| 0.95 | 0.278 | 0.485 | 0.132 | 91% |

Flagging every patient would give precision 0.461.

### Held-out test set (final model, threshold 0.352)

71 patients, 33 admitted to ICU. 95% intervals from 2,000 bootstrap resamples.

| Metric | Test set | 95% interval | Cross-validation |
| --- | --- | --- | --- |
| ROC-AUC | 0.643 | 0.502 – 0.765 | 0.695 |
| PR-AUC | 0.592 | 0.441 – 0.767 | 0.627 |
| Brier score | 0.236 | 0.205 – 0.272 | 0.225 |
| Recall | 0.818 | 0.667 – 0.938 | 0.900 |
| Precision | 0.519 | 0.383 – 0.655 | 0.539 |
| Specificity | 0.342 | 0.200 – 0.500 | 0.340 |
| F1 | 0.635 | 0.507 – 0.739 | — |

At the threshold the model caught **27 of 33** ICU patients, missed 6, and flagged 52 of 71 patients (73%). For reference, the Brier score of always predicting the test set's ICU rate is 0.249.

**Reading these results:** the test results agree with cross-validation within their (wide) intervals, at the lower end. With 71 test patients, the ROC-AUC interval reaches down to about 0.5. Data from the first two hours carries limited signal: the model is useful as a cautious screen that rules out about a third of patients who will not need ICU, not as a confident predictor.

> Earlier versions of this README reported a test ROC-AUC of 0.72 and recall of 0.82 at a 0.5 cut-off. Those figures came from a dataset that included 32 patients already in ICU during the 0–2 hour window (see [Final Modeling Dataset](#final-modeling-dataset)), and the model was chosen on the same test set it was scored on.

---

## Model Selection

**Random Forest with sigmoid calibration**, chosen from cross-validation before the test set was used:

* Discrimination is statistically indistinguishable across candidates, so it does not decide the choice
* Best calibration slope among the calibrated models (1.17), with a Brier score close to the best (0.225 vs 0.222)
* Precision and specificity at each recall target are level with XGBoost + sigmoid
* Needs only scikit-learn at inference time

Choosing among six candidates on the same cross-validation slightly flatters the winner; the single test-set evaluation is the unbiased check.

---

## Key Insights

What the model relies on most (impurity-based importances, summed per measurement; see [`reports/evaluation_test.md`](reports/evaluation_test.md)):

* Age (percentile group and over-65 flag)
* Gender
* Lymphocytes
* C-reactive protein (PCR)
* Platelets
* Urea
* Neutrophils

These are associations the model uses, not causal effects; they are consistent with commonly reported markers of COVID-19 severity.

Two data findings shaped the work:

* **32 patients were already in ICU during the first window.** Their measurements may have been taken after admission, so they were excluded.
* **At 0–2 hours, each lab is effectively one value.** Its MEAN, MEDIAN, MIN and MAX columns are identical copies and its DIFF column is constant, so the 192 columns the model keeps are 36 lab measurements (five columns each) plus 12 demographic and comorbidity fields. The API fills all copies from the single value a user enters.

---

## Engineering and Reproducibility

* Modular Python package (`src/`) with central configuration (`src/config.py`)
* Dataset built in code from the raw file (`src/build_dataset.py`), with the 0–2 hour exclusion rule tested
* All preprocessing inside scikit-learn pipelines, refitted in every cross-validation fold
* Fixed seeds throughout: re-running the pipeline reproduces the same model file byte for byte
* Model saved to a fixed path (`models/icu_model.joblib`) with `models/model_metadata.json`: decision threshold, feature columns, cross-validation and test results, library versions
* The API and dashboard read the threshold from that metadata, so labels always match the evaluated model
* pytest tests (dataset build, preprocessing, threshold and calibration helpers, model loading, API contract), run by GitHub Actions on Python 3.12 and 3.13 for every push and pull request

---

## Deployment Architecture

The application was extended into a deployable end-to-end ML system using:

* FastAPI backend
* Streamlit frontend
* Docker containerization
* Render cloud deployment

### Backend

The FastAPI backend provides:

* `POST /predict`: calibrated probability, flag and threshold
* `GET /model-info`: threshold and held-out test results
* Validated inputs (allowed age groups, 0/1 flags, values within the provider's −1 to 1 scale)
* Inference adapter that fills a lab's identical summary columns from one value
* Automatic Swagger API documentation (local only; the deployed container exposes only the dashboard)

### Frontend

The Streamlit frontend provides:

* Interactive ICU risk dashboard
* Inputs limited to what the model uses (age group, comorbidities, five lab values)
* Calibrated probability and a flag based on the model's saved threshold
* An explanation of the threshold and its test-set performance
* Deployment-safe API configuration

### Inference Flow

```text
User Input
    ↓
Streamlit Frontend
    ↓
FastAPI Backend
    ↓
Inference Adapter
    ↓
Calibrated model + saved threshold
    ↓
ICU risk estimate and flag
```

---

## Dockerization

The application is fully Dockerized for reproducible deployment.

Dockerization provides:

* Portable runtime environment
* Dependency consistency
* Cloud deployment readiness
* Reproducible infrastructure

The Docker container runs:

* FastAPI backend
* Streamlit frontend

inside a single deployment-ready environment.

---

## Cloud Deployment

The application is publicly deployed on Render using a Docker-based deployment workflow.

Deployment stack:

* Render
* Docker
* FastAPI
* Streamlit

GitHub integration enables automatic redeployment on every push to the `main` branch.

---

## Project Structure

```text
icu-covid19-prediction/
├── app/
│   ├── api/main.py              # FastAPI: /predict, /model-info
│   ├── frontend/                # Streamlit dashboard
│   └── utils/inference.py       # user input -> training schema
├── data/
│   ├── raw/                     # source file from Kaggle + README
│   └── processed/df_model.csv   # built by src/build_dataset.py
├── models/
│   ├── icu_model.joblib         # final calibrated model
│   └── model_metadata.json      # threshold, columns, CV and test results
├── notebooks/                   # original exploration (see note below)
├── reports/
│   ├── cv_report.md             # model comparison, calibration, thresholds
│   ├── evaluation_test.md       # final test-set evaluation
│   └── figures/
├── results/                     # CV results (JSON, per-fold CSV)
├── src/
│   ├── config.py                # paths, seeds, final model, recall target
│   ├── build_dataset.py         # raw file -> patient-level dataset
│   ├── preprocessing.py         # pipeline incl. missing-column filter
│   ├── models.py                # candidate models, train/test split
│   ├── cross_validate.py        # repeated CV on the training split
│   ├── threshold.py             # recall-targeted threshold
│   ├── calibration.py           # Brier score, slope, reliability curves
│   ├── evaluate.py              # test-set metrics, bootstrap intervals
│   ├── train.py                 # final model, threshold, test evaluation
│   └── plots.py
├── tests/                       # pytest suite
├── .github/workflows/tests.yml  # CI
├── run.py                       # runs src/train.py
├── requirements.txt             # runtime only (Docker)
├── requirements-train.txt       # + training, plotting, notebooks
├── requirements-dev.txt         # + pytest, httpx2
├── Dockerfile
├── start.sh
└── LICENSE
```

The notebooks are the original exploration. Their dataset step keeps the 32 patients already in ICU during the first window; `src/build_dataset.py` supersedes it.

---

## Local Development

### Clone repository

```bash
git clone https://github.com/Firas-Armoush-ai/icu-covid19-prediction.git
cd icu-covid19-prediction
```

### Create an environment and install dependencies

Tested with Python 3.13 (local) and 3.12 (Docker); CI runs both.

```bash
conda create -n icu-covid19 python=3.13 -y
conda activate icu-covid19
pip install -r requirements-dev.txt
```

Three requirement files, each building on the previous one:

* `requirements.txt`: what the deployed app needs (the Docker image installs only this)
* `requirements-train.txt`: adds XGBoost, plotting and notebook packages for training and evaluation
* `requirements-dev.txt`: adds pytest

### Rebuild the modelling dataset (optional)

`data/processed/df_model.csv` is committed, so this is only needed after changing `src/build_dataset.py`:

```bash
python -m src.build_dataset
```

### Compare models (cross-validation)

```bash
python -m src.cross_validate     # about 2 minutes; writes reports/cv_report.md
```

### Train the final model and evaluate it on the test set

```bash
python run.py                    # about 1 minute; writes models/ and reports/evaluation_test.md
```

The final model and recall target are set in `src/config.py`. Seeds are fixed, so both commands reproduce the committed results.

### Run the tests

```bash
pytest
```

---

## Run Application Locally

### Run FastAPI backend

```bash
uvicorn app.api.main:app --reload
```

### Run Streamlit frontend

```bash
streamlit run app/frontend/streamlit_app.py
```

### Open locally

Frontend:
http://localhost:8501

API docs:
http://127.0.0.1:8000/docs

---

## Docker Usage

### Build image

```bash
docker build -t icu-prediction-app .
```

### Run container

```bash
docker run -p 8501:8501 -p 8000:8000 icu-prediction-app
```

---

## Licence

* **Code** is released under the [MIT License](LICENSE). The MIT License does not cover the data.
* **Data** (`data/`) comes from Hospital Sírio-Libanês and is redistributed under its original [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licence: you may share and adapt it for **non-commercial** purposes, with attribution (see [Dataset source](#dataset-source)).
  * `data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx` is the original file from Kaggle, unmodified (SHA-256 `325889ae…8ebd1`; a test checks it).
  * `data/processed/df_model.csv` is a derived file. Changes made (see `src/build_dataset.py`): kept only the 0–2 hour window (one row per patient), excluded the 32 patients already in ICU in that window, set the target to whether the patient was admitted to ICU in any window, and removed the `WINDOW` column.

This repository does not claim ownership of the dataset.

---

## Limitations

* **Small sample.** 353 patients, with 71 in the test set, so test metrics have wide intervals (ROC-AUC 0.50–0.77).
* **One hospital group, early 2020.** Hospital Sírio-Libanês sites in São Paulo and Brasília, 28 January – 16 April 2020: before vaccines, later variants and changes in treatment. Performance elsewhere or today is unknown.
* **Modest signal in the first two hours.** At the 90% recall target the model still flags about three in four patients.
* **Pre-scaled inputs.** The provider scaled every numeric value to −1 to 1 and did not publish the scaling, so the model cannot take raw clinical values (for example urea in mg/dL), and GENDER is a 0/1 code whose meaning is not documented.
* **Demo inputs are partial.** The dashboard sets age group, gender, two comorbidity flags and five lab values; the model's other inputs are filled with typical values from the training data.
* **No vital signs.** About 70% of patients had none recorded in the first two hours, so the model drops them.
* **Not a clinical tool.** Educational and portfolio project; it has not been validated for clinical use and must not be used to make care decisions.

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost (compared in cross-validation; not used by the final model)

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Testing and CI

* pytest
* GitHub Actions

### Infrastructure

* Docker
* Render

---

## Author

**Firas Armoush**
AI / Machine Learning Engineer

GitHub: https://github.com/Firas-Armoush-ai
Email: [firas.armoush.ai@gmail.com](mailto:firas.armoush.ai@gmail.com)

---
