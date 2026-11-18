"""Page registry, shared by app.py and pages that link to other pages."""

import streamlit as st

from views import method, performance, portfolio, predict

PREDICT = st.Page(predict.render, title="Score a customer", icon=":material/person_search:", url_path="predict",
                  default=True)
PORTFOLIO = st.Page(portfolio.render, title="Score a customer list", icon=":material/groups:", url_path="batch")
PERFORMANCE = st.Page(performance.render, title="Model performance", icon=":material/fact_check:",
                      url_path="performance")
METHOD = st.Page(method.render, title="Data & method", icon=":material/dataset:", url_path="method")

PAGES = [PREDICT, PORTFOLIO, PERFORMANCE, METHOD]
