# app/frontend/streamlit_app.py
# Entry point of the dashboard: page navigation and the sidebar every page shares.
#
# Run from the project root:
#     streamlit run app/frontend/streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="ICU Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

page = st.navigation([
    st.Page("predict.py", title="Risk estimate", icon="🩺", default=True),
    st.Page("evaluation.py", title="How the model was evaluated", icon="📊"),
])

# =========================================================
# SIDEBAR (shared)
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
    to the range -1 to +1, so the inputs are scaled
    values, not clinical units.
    """
)

st.sidebar.markdown(
    """
    **Built by Firas Armoush**
    [GitHub](https://github.com/Firas-Armoush-ai/icu-covid19-prediction)
    """
)

st.sidebar.divider()

page.run()
