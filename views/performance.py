"""How good is the model? Test-set results, the threshold trade-off, calibration and what drives the scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.schema import LABELS
from lib import charts, data, ui


def render() -> None:
    m = data.model()
    ref = data.reference()
    test = data.test_set()
    y, p = test["churned"].to_numpy(), test["proba"].to_numpy()
    meta = m.meta

    ui.header(
        "Model performance · held-out test set",
        "How good is the model?",
        f"Every number on this page comes from <b>{len(y):,} customers the model never saw</b> during training or "
        "tuning. They were set aside before any modelling and scored once at the end.",
        page_icon="fact_check",
        chips=[("model_training", meta["model"]), ("rule", f"Threshold {m.threshold:.2f}"),
               ("dataset", f"{meta['train_rows']:,} training · {len(y):,} test customers")],
    )

    tst = meta["test"]
    c = st.columns(4)
    c[0].metric("Recall", f"{tst['Recall']:.1%}", help="Share of the customers who churned that the model flags.")
    c[1].metric("Precision", f"{tst['Precision']:.1%}", help="Share of flagged customers who really churned.")
    c[2].metric("PR-AUC", f"{tst['PR-AUC']:.3f}", help="Ranking quality across all thresholds. 1.0 is perfect; "
                f"a random ranking would score {y.mean():.2f}.")
    c[3].metric("ROC-AUC", f"{tst['ROC-AUC']:.3f}")

    ui.note(
        f"<b>The model finds {tst['Recall']:.0%} of the customers who closed their card, and {tst['Precision']:.0%} of "
        "the customers it flags really did.</b> Churners are the positive class, the threshold was set on "
        "cross-validated predictions, and the test customers were used for nothing but this final score."
    )

    ui.section("Choose the trade-off", "How many calls does it take to reach how many churners?")
    t = st.slider("Decision threshold", 0.05, 0.95, float(m.threshold), 0.01, format="%.2f",
                  help=f"Recommended: {m.threshold:.2f}, the threshold with the best F2 score in cross-validation.")
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    per = 1000 / len(y)
    ui.insight(
        f"At {t:.2f}, for every <b>1,000 customers</b> the team calls <b>{(tp + fp) * per:.0f}</b> and reaches "
        f"<b>{tp * per:.0f} of the {(tp + fn) * per:.0f}</b> who would leave ({tp / (tp + fn):.0%} recall). "
        f"{fp * per:.0f} calls go to customers who would have stayed ({tp / max(tp + fp, 1):.0%} precision), and "
        f"{fn * per:.0f} churners are missed."
    )
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        ui.chart(charts.confusion(tn, fp, fn, tp), key="cm")
    with b:
        curve = pd.DataFrame({"threshold": np.round(np.arange(0.02, 0.99, 0.01), 2)})
        curve["Recall"] = [((p >= x) & (y == 1)).sum() / y.sum() for x in curve["threshold"]]
        curve["Precision"] = [((p >= x) & (y == 1)).sum() / max((p >= x).sum(), 1) for x in curve["threshold"]]
        ui.chart(charts.tradeoff(curve, t), key="tradeoff")

    ui.section("Can the probabilities be taken literally?", "Do customers churn at the rate the model predicts?")
    bins = pd.cut(p, [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1], include_lowest=True)
    cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        ui.chart(charts.calibration(cal["predicted"].tolist(), cal["observed"].tolist(), cal["n"].tolist()),
                 key="cal")
    with b:
        ui.insight(
            "<b>At the ends, exactly; in the middle, the model is cautious.</b> Each point is a group of test "
            "customers with similar scores (hover for its size). 85% of customers score under 2% or over 95%, and "
            "those groups churn at almost exactly their predicted rate. The few customers in between churn more "
            "often than scored (those scored around 60% churned at about 80%), so a middle score is a lower bound. "
            "Everyone above the 13% threshold gets a call anyway, so this does not change who is flagged. Resampling "
            "and class weights were avoided because they would inflate every probability."
        )
        ui.cap(f"Brier score on the test set: {np.mean((p - y) ** 2):.3f} (always predicting the {y.mean():.0%} "
               f"churn rate would score {np.mean((y.mean() - y) ** 2):.3f}).")

    ui.section("What drives the scores", "Which facts move the churn probability most, across all test customers?")
    imp = pd.Series(ref["importance"]).sort_values(ascending=False)
    ui.chart(charts.importance([LABELS[c] for c in imp.index], imp.round(3).tolist()), key="imp")
    ui.insight(
        "<b>Behaviour, not demographics.</b> How many transactions a customer makes and how much they spend dominate, "
        "followed by revolving balance, the Q4 to Q1 change in activity and the number of products held. A typical "
        "churner made 43 transactions in the year (stayers: 71) and carried no revolving balance (stayers: about "
        "1,400). Income, education and card type barely move the score."
    )
    ui.footer()
