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

The dataset contains clinical and laboratory measurements for **385 confirmed COVID-19 patients**, each observed across multiple temporal windows:

* 0–2 hours
* 2–4 hours
* 4–6 hours
* 6–12 hours
* Above 12 hours

### Final Modeling Dataset

To ensure clinically meaningful prediction and prevent temporal leakage:

* Only the first observation window (0–2 hours) is used
* Each patient is represented once
* The target reflects future ICU admission

The processed dataset (`df_model.csv`) represents a leakage-safe, patient-level dataset used for model training and inference.

### Final Dataset Characteristics

* 385 patients
* 192 engineered input features after preprocessing and encoding
* Balanced ICU target distribution

> Note: `df_model.csv` shape is `(385, 194)` including:
>
> * 192 input features
> * 1 target column
> * 1 patient identifier column

---

## Methodology

### Data Strategy

The project applies a clinically realistic early prediction strategy:

* Restricted analysis to the first observation window (0–2 hours)
* Constructed a patient-level ICU target
* Removed post-admission information
* Prevented temporal and repeated-patient leakage

### Preprocessing Pipeline

The preprocessing workflow was implemented using reusable Scikit-learn pipelines.

#### Numerical Features

* Median imputation
* Standard scaling

#### Categorical Features

* Most frequent imputation
* One-hot encoding

### Feature Engineering

* Removed features with >60% missing values
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
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 02_eda_and_data_understanding.ipynb
│   └── 03_model_experiments.ipynb
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
git clone git@github-ai:Firas-Armoush-ai/icu-covid19-prediction.git
cd icu-covid19-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
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

## Dataset License

Dataset license:

```text
CC BY-NC 4.0
```

This repository does not claim ownership of the original dataset.

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
