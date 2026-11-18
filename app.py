"""
Bank credit card churn prediction: score a customer, rank a customer list and see how good the model is.
The trained pipeline ships in model/ (built by train.py); the inference code is in churn/.
"""

import streamlit as st

st.set_page_config(
    page_title="Credit card churn prediction | Ahmad Hassan",
    page_icon=":material/credit_card:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from lib.ui import inject_css  # noqa: E402
from nav import PAGES  # noqa: E402

inject_css()
st.navigation(PAGES, position="top").run()
