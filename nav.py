"""Page registry (custom console bar, native navigation hidden), shared by app.py and pages that link to other pages."""

import streamlit as st

from views import method, performance, portfolio, predict

PREDICT = st.Page(predict.render, title="Track a customer", icon=":material/radar:", url_path="predict", default=True)
PORTFOLIO = st.Page(portfolio.render, title="Strip board", icon=":material/view_agenda:", url_path="batch")
PERFORMANCE = st.Page(performance.render, title="Calibration", icon=":material/tune:", url_path="performance")
METHOD = st.Page(method.render, title="Manual", icon=":material/menu_book:", url_path="method")

PAGES = [PREDICT, PORTFOLIO, PERFORMANCE, METHOD]
