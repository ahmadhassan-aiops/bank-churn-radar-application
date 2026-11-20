"""Calibration: how good the model is on held-out customers, where to set the call line, whether the probabilities
can be taken literally, and which signals drive the scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.schema import LABELS
from scope import charts, data, ui


def render() -> None:
    m = data.model()
    ref = data.reference()
    test = data.test_set()
    y, p = test["churned"].to_numpy(), test["proba"].to_numpy()
    meta, tst = m.meta, m.meta["test"]

    ui.page("Mode 3 · Calibration", "How good is the model?",
            f"Every number on this page comes from <b>{len(y):,} customers the model never saw</b> in training or tuning: "
            "they were set aside before any modelling and scored once at the end.")
    ui.readouts([("Recall", f"{tst['Recall']:.1%}", "churners the model calls"),
                 ("Precision", f"{tst['Precision']:.1%}", "calls that reach a real churner"),
                 ("PR-AUC", f"{tst['PR-AUC']:.3f}", f"ranking quality; random would be {y.mean():.2f}"),
                 ("ROC-AUC", f"{tst['ROC-AUC']:.3f}", "separates churners from stayers"),
                 ("Call line", f"{m.threshold:.2f}", "best F2 in cross-validation")])

    with ui.module("Bench 01", "Where to draw the call line",
                   "Move the line and see, per 1,000 customers, how many calls it takes to reach how many churners."):
        t = st.slider("Call line", 0.05, 0.95, float(m.threshold), 0.01, format="%.2f",
                      help=f"Recommended: {m.threshold:.2f}, the best F2 score (recall weighed twice) in cross-validation.")
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        per = 1000 / len(y)
        ui.freq(f"At <b>{t:.2f}</b>, for every 1,000 customers the team makes <b>{(tp + fp) * per:.0f}</b> calls and reaches "
                f"<b>{tp * per:.0f}</b> of the <b>{(tp + fn) * per:.0f}</b> who would leave ({tp / max(tp + fn, 1):.0%} recall). "
                f"<b>{fp * per:.0f}</b> calls go to customers who would have stayed ({tp / max(tp + fp, 1):.0%} precision), and "
                f"<b>{fn * per:.0f}</b> churners are missed.")
        a, b = st.columns([1, 1.2], gap="medium")
        with a:
            ui.detection(tp, fp, fn, tn)
        with b:
            curve = pd.DataFrame({"threshold": np.round(np.arange(0.02, 0.99, 0.01), 2)})
            curve["Recall"] = [((p >= x) & (y == 1)).sum() / y.sum() for x in curve["threshold"]]
            curve["Precision"] = [((p >= x) & (y == 1)).sum() / max((p >= x).sum(), 1) for x in curve["threshold"]]
            st.plotly_chart(charts.tradeoff(curve, t), config=charts.CONFIG, key="tradeoff")

    a, b = st.columns([1.15, 1], gap="medium")
    with a, ui.module("Bench 02", "Can the probabilities be taken literally?",
                      "Held-out customers grouped by score: does each group churn at the rate the model predicts? Hover a "
                      "point for its size."):
        bins = pd.cut(p, [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1], include_lowest=True)
        cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
            predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        st.plotly_chart(charts.calibration(cal["predicted"].tolist(), cal["observed"].tolist(), cal["n"].tolist()),
                        config=charts.CONFIG, key="cal")
    with b, ui.module("Reading 02", "What the calibration shows"):
        ui.note("<b>At the ends, exactly; in the middle, cautious.</b> 85% of customers score under 2% or over 95%, and those "
                "groups churn at almost exactly their predicted rate.",
                "The few customers in between churn more often than scored (those scored around 60% churned at about 80%), "
                "so a middle score is a lower bound. Everyone above the 13% call line gets a call anyway, so this does not "
                "change who is called.",
                "Resampling and class weights were not used because they would inflate every probability.")
        ui.readouts([("Brier score", f"{np.mean((p - y) ** 2):.3f}", "lower is better"),
                     ("Base-rate Brier", f"{np.mean((y.mean() - y) ** 2):.3f}", f"always predicting {y.mean():.0%}")])

    with ui.module("Bench 03", "What drives the scores",
                   "Average size of each signal's effect on the churn score across all held-out customers (mean |SHAP|)."):
        imp = pd.Series(ref["importance"]).sort_values(ascending=False)
        a, b = st.columns([1.5, 1], gap="medium")
        a.plotly_chart(charts.importance([LABELS[c] for c in imp.index], imp.round(3).tolist()), config=charts.CONFIG, key="imp")
        with b:
            ui.note("<b>Behaviour, not demographics.</b> How many transactions a customer makes and how much they spend "
                    "dominate, followed by revolving balance, the Q4 to Q1 change in activity and the number of products held.",
                    "A typical churner made 43 transactions in the year (stayers: 71) and carried no revolving balance "
                    "(stayers: about $1,400). Income, education and card type barely move the score.")
    ui.footer()
