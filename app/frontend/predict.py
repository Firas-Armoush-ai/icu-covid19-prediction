# app/frontend/predict.py
# Page 1: estimate ICU risk for one patient through the FastAPI backend.

import streamlit as st
import requests

from config import MODEL_INFO_ENDPOINT, PREDICT_ENDPOINT
from utils import AGE_GROUPS, threshold_explanation, yes_no_to_int

st.sidebar.caption(
    "🧪 Default values are one patient from the training "
    "data (first two hours), who was later admitted to ICU."
)

# =========================================================
# MODEL INFO (threshold and test results, from the API)
# =========================================================

@st.cache_data(ttl=3600)
def load_model_info():
    response = requests.get(MODEL_INFO_ENDPOINT, timeout=10)
    response.raise_for_status()
    return response.json()

# =========================================================
# MAIN TITLE
# =========================================================

st.title("🩺 ICU Risk Prediction Dashboard")

st.markdown(
    """
    Estimate the probability of ICU admission
    from early-stage patient data.
    """
)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Patient Information")

with st.form("prediction_form"):

    # =====================================================
    # DEMOGRAPHICS
    # =====================================================

    st.markdown("### Demographics")

    age_group = st.selectbox(
        "Age group (percentile)",
        AGE_GROUPS,
        index=AGE_GROUPS.index("70th"),
        help=(
            "The dataset records age only as percentile groups, "
            "not years. Groups from the 60th percentile up are "
            "the patients aged over 65."
        )
    )

    gender = st.radio(
        "Gender (dataset code)",
        [0, 1],
        format_func=lambda code: f"Code {code}",
        horizontal=True,
        help=(
            "The dataset provider codes gender as 0 or 1 "
            "without saying which is which."
        )
    )

    htn = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    immuno = st.selectbox(
        "Immunocompromised",
        ["No", "Yes"]
    )

    st.divider()

    # =====================================================
    # LABORATORY VALUES
    # =====================================================

    st.markdown("### Laboratory Measurements")

    st.caption(
        "Scaled laboratory values (-1 to +1, as published by the "
        "data provider). These are the five lab results the model "
        "relies on most; its other lab results are filled with typical "
        "values from the training data."
    )

    def lab_slider(label, default):
        return st.slider(
            label,
            min_value=-1.0,
            max_value=1.0,
            value=default,
            step=0.01
        )

    linfocitos = lab_slider("Lymphocytes", -0.803)
    pcr = lab_slider("C-reactive protein (PCR)", -0.791)
    platelets = lab_slider("Platelets", -0.704)
    urea = lab_slider("Urea", -0.836)
    neutrophils = lab_slider("Neutrophils", -0.853)

    st.caption(
        "Vital signs are not inputs: about 70% of patients had none "
        "recorded in the first two hours, so the model does not use them."
    )

    st.divider()

    submitted = st.form_submit_button(
        "Estimate ICU Risk"
    )

# =========================================================
# PREDICTION
# =========================================================

if submitted:

    payload = {
        "AGE_PERCENTIL": age_group,
        "GENDER": gender,
        "HTN": yes_no_to_int(htn),
        "IMMUNOCOMPROMISED": yes_no_to_int(immuno),
        "LINFOCITOS_MEDIAN": linfocitos,
        "PCR_MEDIAN": pcr,
        "PLATELETS_MEDIAN": platelets,
        "UREA_MEDIAN": urea,
        "NEUTROPHILES_MEDIAN": neutrophils,
    }

    try:

        with st.spinner("Estimating..."):

            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=10
            )

        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            result = response.json()

            probability = result["icu_probability"]

            st.divider()

            st.subheader("Result")

            if result["prediction"] == 1:
                st.error(result["risk_level"])
            else:
                st.success(result["risk_level"])

            st.metric(
                label="Estimated probability of ICU admission",
                value=f"{probability:.0%}"
            )

            st.progress(float(probability))

            # ---------------------------------------------
            # How to read the result
            # ---------------------------------------------

            st.markdown("### How to read this")

            try:
                st.markdown(threshold_explanation(load_model_info()))
            except requests.exceptions.RequestException:
                st.markdown(
                    f"Patients at or above **{result['threshold']:.0%}** are flagged."
                )

            st.write(
                "The probability is calibrated: among many patients given "
                "a similar estimate, about that share went on to need ICU. "
                "Data from the first two hours carries limited signal, so "
                "most estimates fall between about 25% and 70%."
            )

        # =================================================
        # API ERROR
        # =================================================

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.json(response.json())

    # =====================================================
    # CONNECTION ERROR
    # =====================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI backend.\n"
            "Ensure the API server is running."
        )

    # =====================================================
    # GENERIC ERROR
    # =====================================================

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )
