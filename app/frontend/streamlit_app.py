import streamlit as st
import requests

from config import MODEL_INFO_ENDPOINT, PREDICT_ENDPOINT
from utils import AGE_GROUPS, threshold_explanation, yes_no_to_int

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ICU Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("About")

st.sidebar.markdown(
    """
    This application estimates the risk of ICU admission
    for confirmed COVID-19 patients from data collected
    in the first two hours after admission.

    Built with:
    - Streamlit
    - FastAPI
    - scikit-learn (calibrated random forest)
    - Docker
    """
)

st.sidebar.divider()

st.sidebar.warning(
    """
    **Not a clinical tool.** Educational and portfolio
    project, built from 353 patients at one hospital
    group in Brazil (January–April 2020).

    The dataset provider scaled every numeric value
    to the range -1 to +1, so the inputs below are
    scaled values, not clinical units.
    """
)

st.sidebar.caption(
    "🧪 Default values are an example patient profile "
    "from the dataset."
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Built by Firas Armoush**
    [GitHub](https://github.com/Firas-Armoush-ai/icu-covid19-prediction)
    """
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
        "data provider). These five are sent as entered; the model's "
        "other lab results are filled with typical values from the "
        "training data."
    )

    albumin = st.slider(
        "Albumin",
        min_value=-1.0,
        max_value=1.0,
        value=0.263,
        step=0.01
    )

    pcr = st.slider(
        "C-reactive protein (PCR)",
        min_value=-1.0,
        max_value=1.0,
        value=-0.958,
        step=0.01
    )

    urea = st.slider(
        "Urea",
        min_value=-1.0,
        max_value=1.0,
        value=-0.880,
        step=0.01
    )

    creatinin = st.slider(
        "Creatinine",
        min_value=-1.0,
        max_value=1.0,
        value=-0.924,
        step=0.01
    )

    be_arterial = st.slider(
        "Arterial base excess",
        min_value=-1.0,
        max_value=1.0,
        value=-1.0,
        step=0.01
    )

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
        "ALBUMIN_MEDIAN": albumin,
        "BE_ARTERIAL_MEDIAN": be_arterial,
        "PCR_MEDIAN": pcr,
        "UREA_MEDIAN": urea,
        "CREATININ_MEDIAN": creatinin,
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
