"""
Bank Churn Radar: credit card churn on a radar console. Track one customer on the scope, sort a customer list into
call / watch / clear bays, and check the model's calibration. Trained pipeline in model/ (built by train.py),
inference code in churn/.
"""

import base64
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Bank Churn Radar | Ahmad Hassan", page_icon=":material/radar:", layout="wide",
                   initial_sidebar_state="collapsed")

from nav import PAGES  # noqa: E402
from scope import data  # noqa: E402
from scope.ui import inject_css, status_line  # noqa: E402

LOGO = base64.b64encode((Path(__file__).parent / "assets" / "logo.svg").read_bytes()).decode()

inject_css()
current = st.navigation(PAGES, position="hidden")
cols = st.columns([1.55, 1.05, 0.95, 0.95, 0.8], vertical_alignment="center")
cols[0].html(f'<div class="cbar"><img src="data:image/svg+xml;base64,{LOGO}" alt=""><div><p class="nm">CHURN <b>RADAR</b></p>'
             '<p class="sub">Credit card retention console</p></div></div>')
for col, p in zip(cols[1:], PAGES):
    col.page_link(p, label=p.title, icon=p.icon, width="stretch")
k = [p.title for p in PAGES].index(current.title) + 2
st.html(f'<style>[data-testid="stHorizontalBlock"]:has(.cbar) > [data-testid="stColumn"]:nth-child({k}) a '
        '{ border-color: var(--scope); background: var(--scope-soft); box-shadow: inset 0 -3px 0 var(--scope); }</style>')
m = data.model()
test = m.meta["test"]
status_line([("Model online", m.meta["model"]), ("Call line", f"{m.threshold:.0%}"), ("Recall", f"{test['Recall']:.0%}"),
             ("Precision", f"{test['Precision']:.0%}"), ("PR-AUC", f"{test['PR-AUC']:.3f}"),
             ("Held-out customers", f"{m.meta['test_rows']:,}")])
current.run()
