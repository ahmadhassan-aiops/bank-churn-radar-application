"""Track a customer: the customer file on the left; on the right the customer on the radar scope among the held-out
customers and the risk readout; then the signals that drive it and the retention orders. Updates live."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from churn import actions
from churn.model import to_prob
from churn.schema import CARDS, EDUCATION, GENDERS, INCOME, LABELS, MARITAL, Customer
from scope import charts, data, ui

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
        f"Loaded test customer <b>#{int(row['row_id'])}</b>, who in reality "
        f"<b>{'closed their card' if row['churned'] else 'stayed'}</b>. The model never saw this customer in training."
    )


def _init() -> None:
    if KEY + "Customer_Age" not in st.session_state:
        _load(DEFAULTS)
        st.session_state["example_note"] = "Tracking a typical customer (the median of every field). Change anything in the file."


def _group(label: str) -> None:
    st.html(f'<p class="mode" style="margin:.5rem 0 .1rem">{label}</p>')


def _form() -> dict:
    v = {}
    k = lambda c: {"key": KEY + c, "label": LABELS[c]}  # noqa: E731
    _group("Who they are")
    a, b = st.columns(2)
    v["Customer_Age"] = a.number_input(min_value=18, max_value=90, step=1, **k("Customer_Age"))
    v["Gender"] = b.segmented_control(options=GENDERS, format_func={"F": "Female", "M": "Male"}.get, required=True,
                                      **k("Gender"))
    v["Dependent_count"] = st.slider(min_value=0, max_value=5, **k("Dependent_count"))
    a, b = st.columns(2)
    v["Education_Level"] = a.selectbox(options=EDUCATION, **k("Education_Level"))
    v["Marital_Status"] = b.selectbox(options=MARITAL, **k("Marital_Status"))
    v["Income_Category"] = st.selectbox(options=INCOME, **k("Income_Category"))
    _group("Relationship with the bank")
    a, b = st.columns(2)
    v["Card_Category"] = a.selectbox(options=CARDS, **k("Card_Category"))
    v["Months_on_book"] = b.number_input(min_value=0, max_value=120, step=1, **k("Months_on_book"))
    v["Total_Relationship_Count"] = st.slider(min_value=1, max_value=6, **k("Total_Relationship_Count"))
    v["Months_Inactive_12_mon"] = st.slider(min_value=0, max_value=12, **k("Months_Inactive_12_mon"))
    v["Contacts_Count_12_mon"] = st.slider(min_value=0, max_value=12, **k("Contacts_Count_12_mon"))
    _group("Card activity, last 12 months")
    a, b = st.columns(2)
    v["Credit_Limit"] = a.number_input(min_value=100.0, max_value=100_000.0, step=500.0, format="%.0f", **k("Credit_Limit"))
    v["Total_Revolving_Bal"] = b.number_input(min_value=0.0, max_value=100_000.0, step=100.0, format="%.0f",
                                              **k("Total_Revolving_Bal"))
    a, b = st.columns(2)
    v["Total_Trans_Ct"] = a.number_input(min_value=0, max_value=500, step=1, **k("Total_Trans_Ct"))
    v["Total_Trans_Amt"] = b.number_input(min_value=0.0, max_value=100_000.0, step=100.0, format="%.0f", **k("Total_Trans_Amt"))
    a, b = st.columns(2)
    v["Total_Ct_Chng_Q4_Q1"] = a.number_input(min_value=0.0, max_value=10.0, step=0.05, format="%.2f",
                                              help="Q1 transactions divided by Q4 transactions. Below 1 means fewer.",
                                              **k("Total_Ct_Chng_Q4_Q1"))
    v["Total_Amt_Chng_Q4_Q1"] = b.number_input(min_value=0.0, max_value=10.0, step=0.05, format="%.2f",
                                               help="Q1 spend divided by Q4 spend. Below 1 means less.", **k("Total_Amt_Chng_Q4_Q1"))
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
    test = data.test_set()
    ui.page("Mode 1 · Track a customer", "Will this customer close their card?",
            "Fill in the customer file. The model places the customer on the scope among the bank's held-out "
            f"customers, gives the <b>chance that they churn</b>, calls them in at <b>{m.threshold:.0%} or more</b>, and "
            "shows <b>which signals drive the score</b> and what the retention team should do.")

    b = st.columns(4)
    b[0].button("Typical customer", icon=":material/person:", on_click=_load, args=(DEFAULTS,), width="stretch")
    b[1].button("At-risk example", icon=":material/warning:", on_click=_load_example, args=("risk",), width="stretch")
    b[2].button("Loyal example", icon=":material/verified:", on_click=_load_example, args=("loyal",), width="stretch")
    b[3].button("Random customer", icon=":material/casino:", help="Load a random customer from the held-out set",
                on_click=_load_example, args=("random",), width="stretch")
    if note := st.session_state.get("example_note"):
        ui.cap(note)

    left, right = st.columns([1, 1.3], gap="medium")
    with left, ui.module("File 01", "Customer file", "Every field the model reads. The scope updates as you type."):
        values = _form()
    try:
        customer = Customer(**values)
    except ValidationError as exc:
        with right:
            st.error(exc.errors()[0]["msg"].removeprefix("Value error, "), icon=":material/error:")
        return
    row = customer.to_frame()
    p = float(m.predict_proba(row)[0])
    band = m.band(p)
    contrib, bias = m.explain(row)
    contrib = contrib.iloc[0]
    row0 = row.iloc[0]
    stay = data.reference()["median_stayed"]
    t = m.threshold
    share_below = float((test["proba"] < p).mean())
    text = {"Flag": f"<b>Call this customer.</b> Riskier than {share_below:.0%} of the bank's customers; a retention call costs "
                    "little next to a lost card.",
            "Watch": "<b>Inside the watch ring.</b> Below the call line; re-score after next quarter's activity.",
            "Low": "<b>Clear.</b> Unlikely to churn; no retention action is needed now."}[band]

    with right, ui.module("Scope 02", "On the radar",
                          "Each dot is a held-out customer, red if they really churned. <b>The closer to the centre, the "
                          "likelier to churn</b>; inside the red ring the team calls. Bearing (clockwise from north) is the "
                          "number of card transactions in the last 12 months.", band=band):
        st.plotly_chart(charts.radar(test["proba"].to_numpy(), test["Total_Trans_Ct"].to_numpy(), t,
                                     outcome=test["churned"].to_numpy(), me=(p, customer.Total_Trans_Ct),
                                     ids=test["row_id"].to_numpy()), config=charts.CONFIG, key="scope")
        ui.lock(p, band, text, [("Low", f"under {t / 2:.1%}"), ("Watch", f"{t / 2:.1%} to {t:.0%}"), ("Flag", f"{t:.0%}+")],
                [("Credit available", f"${customer.open_to_buy:,.0f}"), ("Utilisation", f"{customer.utilisation:.0%}"),
                 ("Riskier than", f"{share_below:.0%} of customers")])

    with right, ui.module("Signals 03", "What drives the score",
                   "The six facts that move this customer's score most (SHAP values, log-odds), next to a typical customer "
                   f"who stayed. From the model's baseline of {to_prob(float(bias[0])):.0%}, they add up to this score."):
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6]
        ui.signals([{"name": LABELS[c], "value": _value(c, row0[c]), "typical": _value(c, stay[c]) if c in stay else "",
                     "shap": float(v)} for c, v in top.items()])
        ui.cap("Factors interact, so the same value can push differently for different customers.")

    with ui.module("Orders 04", "What the retention team should do",
                   "Actions for the signals that raise this customer's risk" if band != "Low"
                   else "Risk is low; these address the signals that raise it most"):
        items = []
        for feature, (ic, title, body) in actions.plan(contrib, row0):
            why = ""
            if feature:
                why = f"{LABELS[feature]}: {_value(feature, row0[feature])}"
                if feature in stay:
                    why += f" (typical stayer {_value(feature, stay[feature])})"
            items.append((ic, title, why, body))
        ui.instructions(items)
    ui.footer()
