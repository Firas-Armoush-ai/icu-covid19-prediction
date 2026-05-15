import streamlit as st
import requests

from config import PREDICT_ENDPOINT
from utils import (
    get_risk_level,
    get_risk_color
)

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
    This application predicts ICU admission risk
    using early-stage COVID-19 clinical data.

    Built with:
    - Streamlit
    - FastAPI
    - Scikit-learn
    - Docker-ready architecture
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Prediction uses the first 0–2 hours "
    "of clinical measurements."
)

st.sidebar.warning(
    """
    This demo uses normalized research dataset features
    and is intended for educational and portfolio
    purposes only.

    Most numerical inputs are scaled approximately
    between -1 and +1.
    """
)

st.sidebar.caption(
    "🧪 Default values represent an example "
    "normalized patient profile from the dataset."
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Built by Firas Armoush**  
    [GitHub](https://github.com/Firas-Armoush-ai)
    """
)
# =========================================================
# MAIN TITLE
# =========================================================

st.title("🩺 ICU Risk Prediction Dashboard")

st.markdown(
    """
    Predict the probability of ICU admission
    using early-stage patient clinical data.
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

    age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=68
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
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
    # VITAL SIGNS
    # =====================================================

    st.markdown("### Vital Signs")

    st.caption(
        "Normalized clinical features "
        "(approximately scaled between -1 and +1)"
    )

    heart_rate = st.slider(
        "HEART_RATE_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-0.358,
        step=0.01,
        help="Normalized heart rate feature"
    )

    respiratory_rate = st.slider(
        "RESPIRATORY_RATE_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-0.586,
        step=0.01,
        help="Normalized respiratory rate feature"
    )

    oxygen_saturation = st.slider(
        "OXYGEN_SATURATION_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=0.684,
        step=0.01,
        help="Normalized oxygen saturation feature"
    )

    st.divider()

    # =====================================================
    # LABORATORY VALUES
    # =====================================================

    st.markdown("### Laboratory Measurements")

    st.caption(
        "Normalized laboratory features "
        "(approximately scaled between -1 and +1)"
    )

    albumin = st.slider(
        "ALBUMIN_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=0.263,
        step=0.01,
        help="Normalized albumin feature"
    )

    pcr = st.slider(
        "PCR_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-0.958,
        step=0.01,
        help="Normalized PCR feature"
    )

    urea = st.slider(
        "UREA_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-0.880,
        step=0.01,
        help="Normalized urea feature"
    )

    creatinin = st.slider(
        "CREATININ_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-0.924,
        step=0.01,
        help="Normalized creatinin feature"
    )

    be_arterial = st.slider(
        "BE_ARTERIAL_MEDIAN",
        min_value=-1.0,
        max_value=1.0,
        value=-1.0,
        step=0.01,
        help="Normalized arterial base excess feature"
    )

    st.divider()

    submitted = st.form_submit_button(
        "Predict ICU Risk"
    )

# =========================================================
# PREDICTION
# =========================================================

if submitted:

    payload = {

        "AGE_ABOVE65": 1 if age >= 65 else 0,

        "GENDER": 1 if gender == "Male" else 0,

        "HTN": 1 if htn == "Yes" else 0,

        "IMMUNOCOMPROMISED": (
            1 if immuno == "Yes" else 0
        ),

        "ALBUMIN_MEDIAN": albumin,

        "BE_ARTERIAL_MEDIAN": be_arterial,

        "PCR_MEDIAN": pcr,

        "UREA_MEDIAN": urea,

        "CREATININ_MEDIAN": creatinin,

        "HEART_RATE_MEDIAN": heart_rate,

        "RESPIRATORY_RATE_MEDIAN": respiratory_rate,

        "OXYGEN_SATURATION_MEDIAN": oxygen_saturation
    }

    try:

        with st.spinner(
            "Generating prediction..."
        ):

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

            probability = result[
                "icu_probability"
            ]

            prediction = result[
                "prediction"
            ]

            risk_level = get_risk_level(
                probability
            )

            risk_color = get_risk_color(
                probability
            )

            st.success(
                "Prediction completed successfully."
            )

            st.divider()

            st.subheader("Prediction Result")

            # ---------------------------------------------
            # Risk label
            # ---------------------------------------------

            if prediction == 1:

                st.error(
                    f"{risk_level}"
                )

            else:

                st.success(
                    f"{risk_level}"
                )

            # ---------------------------------------------
            # Probability
            # ---------------------------------------------

            st.metric(
                label="ICU Risk Probability",
                value=f"{probability:.2%}"
            )

            # ---------------------------------------------
            # Progress bar
            # ---------------------------------------------

            st.progress(float(probability))

            # ---------------------------------------------
            # Interpretation
            # ---------------------------------------------

            st.markdown(
                "### Clinical Interpretation"
            )

            st.write(
                "The prediction reflects the estimated "
                "probability that the patient may "
                "require ICU admission based on "
                "early-stage normalized clinical features."
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