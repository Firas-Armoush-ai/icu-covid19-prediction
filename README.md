# ICU Prediction for COVID-19 Patients (Early-Stage Machine Learning System)

🚀 **Live Demo:**
https://icu-covid19-prediction.onrender.com/

📘 **API Documentation:**
https://icu-covid19-prediction.onrender.com/docs

---

## Overview

This project develops an end-to-end machine learning system to predict whether a patient with confirmed COVID-19 will require Intensive Care Unit (ICU) admission using only early-stage clinical data collected within the first 0–2 hours.

The project evolved from exploratory notebook experimentation into a modular, reproducible, deployment-ready ML application featuring:

* Leakage-safe preprocessing
* Reusable ML pipelines
* FastAPI inference backend
* Streamlit interactive frontend
* Dockerized deployment architecture
* Cloud deployment on Render

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

## Modeling

The following models were evaluated:

* Logistic Regression (baseline)
* Random Forest
* XGBoost

All models were trained using a stratified 80/20 train-test split.

---

## Results

| Model         | Accuracy   | Precision | Recall     | F1         | ROC-AUC    |
| ------------- | ---------- | --------- | ---------- | ---------- | ---------- |
| logistic      | 0.6883     | 0.6829    | 0.7179     | 0.7000     | 0.6920     |
| random_forest | **0.7013** | 0.6667    | **0.8205** | **0.7356** | 0.7203     |
| xgboost       | 0.6494     | 0.6304    | 0.7436     | 0.6824     | **0.7379** |

---

## Model Selection

Random Forest was selected as the final model based on its overall balance between recall and F1-score, which are especially important for this clinical use case.

### Selection Rationale

* Highest F1-score across all evaluated models
* Highest recall (~0.82), reducing false negatives
* Stable overall performance across metrics
* Strong performance on tabular clinical data

Although XGBoost achieved a slightly higher ROC-AUC, Random Forest provided the most suitable trade-off for early ICU risk prediction.

From a clinical perspective, minimizing false negatives (missed ICU cases) is a priority.

---

## Key Insights

The model identified clinically meaningful predictors associated with ICU admission risk, including:

* Age
* Lymphocyte levels
* Urea
* C-reactive protein (PCR)
* Platelets
* Sodium levels

These findings align with known indicators of COVID-19 severity.

---

## Engineering and Reproducibility

The project was refactored from exploratory notebooks into a modular, reproducible, production-oriented ML engineering workflow.

Implemented engineering improvements include:

* Modular Python package structure (`src/`)
* Reusable preprocessing pipelines
* Centralized configuration management
* CLI execution interface (`run.py`)
* Versioned model persistence
* Automated evaluation reporting
* Metrics tracking using JSON artifacts
* Versioned figure generation
* Windows-safe path handling
* Deployment-safe inference architecture
* Reproducible training workflows

The best-performing model is automatically selected, versioned, and saved during training.

---

## Deployment Architecture

The application was extended into a deployable end-to-end ML system using:

* FastAPI backend
* Streamlit frontend
* Docker containerization
* Render cloud deployment

### Backend

The FastAPI backend provides:

* Live inference API
* Schema-safe prediction handling
* Reusable inference adapter layer
* Automatic Swagger API documentation

### Frontend

The Streamlit frontend provides:

* Interactive ICU risk dashboard
* Structured clinical inputs
* Probability visualization
* Risk-level interpretation
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
Saved sklearn Pipeline
    ↓
ICU Risk Prediction
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
│   ├── api/
│   ├── frontend/
│   └── utils/
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_data_understanding.ipynb
│   └── 02_model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── plots.py
├── models/
├── reports/
├── results/
├── Dockerfile
├── .dockerignore
├── start.sh
├── run.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Local Development

### Clone repository

```bash
git clone https://github.com/Firas-Armoush-ai/icu-covid19-prediction.git
cd icu-covid19-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Rebuild the modelling dataset (optional)

`data/processed/df_model.csv` is committed, so this is only needed after changing `src/build_dataset.py`:

```bash
python -m src.build_dataset
```

### Run ML training pipeline

```bash
python run.py
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
  * `data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx` is the source file from Kaggle.
  * `data/processed/df_model.csv` is a derived file. Changes made (see `src/build_dataset.py`): kept only the 0–2 hour window (one row per patient), excluded the 32 patients already in ICU in that window, set the target to whether the patient was admitted to ICU in any window, and removed the `WINDOW` column.

This repository does not claim ownership of the dataset.

---

## Important Note About Inputs

Many numerical features in the research dataset were already normalized by the original dataset providers.

Therefore:

* frontend numerical inputs represent normalized clinical features
* the deployed application is intended for educational and portfolio purposes
* the system should not be interpreted as a real clinical diagnostic tool

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

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
