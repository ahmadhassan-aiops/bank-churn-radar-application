"""Score one customer: a form on the left, the prediction and its reasons on the right. Updates live."""

from __future__ import annotations

import html

import streamlit as st
from pydantic import ValidationError

from churn import actions
from churn.model import to_prob
from churn.schema import CARDS, EDUCATION, GENDERS, INCOME, LABELS, MARITAL, Customer
from lib import charts, data, ui

DEFAULTS = {  # the typical (median / most common) customer in the training data
    "Customer_Age": 46, "Gender": "F", "Dependent_count": 2, "Education_Level": "Graduate",
    "Marital_Status": "Married", "Income_Category": "Less than $40K", "Card_Category": "Blue",
    "Months_on_book": 36, "Total_Relationship_Count": 4, "Months_Inactive_12_mon": 2, "Contacts_Count_12_mon": 2,
    "Credit_Limit": 4550.0, "Total_Revolving_Bal": 1280.0, "Total_Amt_Chng_Q4_Q1": 0.74, "Total_Trans_Amt": 3900.0,
    "Total_Trans_Ct": 67, "Total_Ct_Chng_Q4_Q1": 0.70,
}
KEY = "in_"  # widget keys are KEY + column name


def _load(values: dict) -> None:
    for col, v in values.items():
        if col in DEFAULTS:
            st.session_state[KEY + col] = type(DEFAULTS[col])(v)


def _load_example(kind: str) -> None:
    test = data.test_set()
    if kind == "risk":
        pool = test[(test["churned"] == 1) & test["proba"].between(0.75, 0.95)]
    elif kind == "loyal":
        pool = test[(test["churned"] == 0) & (test["proba"] < 0.02)]
    else:
        pool = test
    row = pool.sample(1).iloc[0].to_dict()
    _load(row)
    st.session_state["example_note"] = (
        f"Loaded test customer #{int(row['row_id'])}, who in reality "
        f"<b>{'closed their card' if row['churned'] else 'stayed'}</b>. The model never saw this customer in training."
    )


def _init() -> None:
    if KEY + "Customer_Age" not in st.session_state:
        _load(DEFAULTS)
        st.session_state["example_note"] = "Showing a typical customer (the median of every field). Change anything."


def _form() -> dict:
    v = {}
    k = lambda c: {"key": KEY + c, "label": LABELS[c]}  # noqa: E731
    with st.container(border=True):
        st.markdown("**Who they are**")
        a, b = st.columns(2)
        v["Customer_Age"] = a.number_input(min_value=18, max_value=90, step=1, **k("Customer_Age"))
        v["Gender"] = b.segmented_control(options=GENDERS, format_func={"F": "Female", "M": "Male"}.get,
                                          required=True, **k("Gender"))
        v["Dependent_count"] = st.slider(min_value=0, max_value=5, **k("Dependent_count"))
        a, b = st.columns(2)
        v["Education_Level"] = a.selectbox(options=EDUCATION, **k("Education_Level"))
        v["Marital_Status"] = b.selectbox(options=MARITAL, **k("Marital_Status"))
        v["Income_Category"] = st.selectbox(options=INCOME, **k("Income_Category"))
    with st.container(border=True):
        st.markdown("**Relationship with the bank**")
        a, b = st.columns(2)
        v["Card_Category"] = a.selectbox(options=CARDS, **k("Card_Category"))
        v["Months_on_book"] = b.number_input(min_value=0, max_value=120, step=1, **k("Months_on_book"))
        v["Total_Relationship_Count"] = st.slider(min_value=1, max_value=6, **k("Total_Relationship_Count"))
        v["Months_Inactive_12_mon"] = st.slider(min_value=0, max_value=12, **k("Months_Inactive_12_mon"))
        v["Contacts_Count_12_mon"] = st.slider(min_value=0, max_value=12, **k("Contacts_Count_12_mon"))
    with st.container(border=True):
        st.markdown("**Card activity, last 12 months**")
        a, b = st.columns(2)
        v["Credit_Limit"] = a.number_input(min_value=100.0, max_value=100_000.0, step=500.0, format="%.0f",
                                           **k("Credit_Limit"))
        v["Total_Revolving_Bal"] = b.number_input(min_value=0.0, max_value=100_000.0, step=100.0, format="%.0f",
                                                  **k("Total_Revolving_Bal"))
        a, b = st.columns(2)
        v["Total_Trans_Ct"] = a.number_input(min_value=0, max_value=500, step=1, **k("Total_Trans_Ct"))
        v["Total_Trans_Amt"] = b.number_input(min_value=0.0, max_value=100_000.0, step=100.0, format="%.0f",
                                              **k("Total_Trans_Amt"))
        a, b = st.columns(2)
        v["Total_Ct_Chng_Q4_Q1"] = a.number_input(min_value=0.0, max_value=10.0, step=0.05, format="%.2f",
                                                  help="Q1 transactions divided by Q4 transactions. Below 1 means fewer.",
                                                  **k("Total_Ct_Chng_Q4_Q1"))
        v["Total_Amt_Chng_Q4_Q1"] = b.number_input(min_value=0.0, max_value=10.0, step=0.05, format="%.2f",
                                                   help="Q1 spend divided by Q4 spend. Below 1 means less.",
                                                   **k("Total_Amt_Chng_Q4_Q1"))
    return v


def _value(col: str, x) -> str:
    if col in ("Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Amt", "Avg_Open_To_Buy"):
        return f"${x:,.0f}"
    if col == "Avg_Utilization_Ratio":
        return f"{x:.0%}"
    if col in ("Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1"):
        return f"{x:.2f}"
    if isinstance(x, float):
        return f"{x:,.0f}"
    return str(x)


def render() -> None:
    _init()
    m = data.model()
    ref = data.reference()
    ui.header(
        "Bank Churn Radar · XGBoost model",
        "Will this customer close their card?",
        f"Enter a customer's profile and card activity. The model, trained on {m.meta['train_rows']:,} customers, gives the "
        f"<b>probability that they churn</b>, flags them for the retention team at <b>{m.threshold:.0%} or more</b>, "
        "and shows <b>which facts drive the score</b> and what to do about them.",
        chips=[("target", f"Catches {ref['test']['Recall']:.0%} of churners"),
               ("verified", f"{ref['test']['Precision']:.0%} of flags are right"),
               ("bolt", "Live: updates as you type")],
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.button("Typical customer", icon=":material/person:", on_click=_load, args=(DEFAULTS,), width="stretch")
    b2.button("At-risk example", icon=":material/warning:", on_click=_load_example, args=("risk",), width="stretch")
    b3.button("Loyal example", icon=":material/verified:", on_click=_load_example, args=("loyal",), width="stretch")
    b4.button("Random test customer", icon=":material/casino:", on_click=_load_example, args=("random",),
              width="stretch")
    if note := st.session_state.get("example_note"):
        st.html(f'<p class="sec-q">{ui.icon("info")}<span>{note}</span></p>')

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        values = _form()
    with right:
        try:
            customer = Customer(**values)
        except ValidationError as exc:
            st.error(exc.errors()[0]["msg"].removeprefix("Value error, "), icon=":material/error:")
            return
        row = customer.to_frame()
        p = float(m.predict_proba(row)[0])
        band = m.band(p)
        contrib, bias = m.explain(row)
        contrib = contrib.iloc[0]

        if band == "Flag":
            riskier = float((data.test_set()["proba"] > p).mean())
            text = (f"<b>Call this customer.</b> Their score is higher than {1 - riskier:.0%} of the bank's customers. "
                    "A retention call costs little next to a lost card.")
        elif band == "Watch":
            text = ("<b>Below the flag threshold, but worth watching.</b> Re-score after next quarter's activity.")
        else:
            text = "<b>Unlikely to churn.</b> No retention action needed now."
        ui.result_card(p, band, m.threshold, text)
        ui.chips([("account_balance_wallet", f"Credit still available ${customer.open_to_buy:,.0f}"),
                  ("percent", f"Utilisation {customer.utilisation:.0%}")])

        st.subheader("Why this score", anchor=False)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
        stay, churn = ref["median_stayed"], ref["median_churned"]
        details = []
        for col in top.index:
            x = row.iloc[0][col]
            if col in stay:
                details.append(f"This customer: {_value(col, x)} · typical stayer {_value(col, stay[col])} "
                               f"· typical churner {_value(col, churn[col])}")
            else:
                details.append(f"This customer: {_value(col, x)}")
        labels = [f"{LABELS[c]}: {_value(c, row.iloc[0][c])}" for c in top.index]
        ui.chart(charts.drivers(labels, top.round(3).tolist(), details), key="drivers")
        ui.cap(f"Each bar is the feature's SHAP contribution for this customer. Starting from the model's baseline "
               f"({to_prob(float(bias[0])):.0%}), the bars add up to this customer's score. "
               "Hover a bar to compare with typical customers.")

        st.subheader("What to do", anchor=False)
        items = actions.suggest(contrib, row.iloc[0])
        st.html('<div class="recs" style="--n:1">' + "".join(
            f'<div class="rec"><div class="rec-head"><span class="rec-num">{n}</span>'
            f'<span class="msr rec-icon" aria-hidden="true">{html.escape(i)}</span></div>'
            f'<p class="rec-title">{html.escape(t)}</p><p class="rec-text">{html.escape(x)}</p></div>'
            for n, (i, t, x) in enumerate(items, 1)) + "</div>")
        if band == "Low":
            ui.cap("The actions address what raises this customer's risk most, even though it is low overall.")

    ui.footer()
