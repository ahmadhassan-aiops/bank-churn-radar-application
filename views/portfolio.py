"""Score a whole list of customers: rank them by risk and plan how many the retention team should call."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn.schema import INPUT_COLUMNS, LABELS, validate_rows
from lib import charts, data, ui

TEST = "The 2,026 held-out test customers (outcomes known)"
UPLOAD = "Upload my own CSV"


@st.cache_data(show_spinner="Scoring customers...")
def _score(rows: pd.DataFrame) -> pd.DataFrame:
    m = data.model()
    proba = m.predict_proba(rows)
    contrib, _ = m.explain(rows)
    top = contrib.idxmax(axis=1).where(contrib.max(axis=1) > 0)
    out = rows.copy()
    out.insert(0, "Churn probability", proba)
    out.insert(1, "Risk", [m.band(p) for p in proba])
    out.insert(2, "Main reason", top.map(LABELS).fillna("Nothing raises the risk"))
    return out


def render() -> None:
    m = data.model()
    ui.header(
        "Batch scoring · retention planning",
        "Who should the retention team call first?",
        "Score a whole customer list at once, rank it by churn risk and see the trade-off: "
        "<b>how many calls it takes to reach how many of the customers who would leave</b>.",
        page_icon="groups",
        chips=[("upload_file", "Your CSV or the test set"), ("sort", "Ranked by risk"), ("download", "CSV export")],
    )

    source = st.segmented_control("Customers to score", [TEST, UPLOAD], default=TEST, required=True)
    outcome = None
    if source == UPLOAD:
        template = data.test_set()[INPUT_COLUMNS].head(5)
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        file = c1.file_uploader("CSV with one row per customer, using the dataset's column names", type="csv")
        c2.download_button("Template (5 rows)", template.to_csv(index=False), "churn_template.csv", "text/csv",
                           icon=":material/download:", width="stretch")
        if file is None:
            ui.cap("Required columns: " + ", ".join(INPUT_COLUMNS) + ". Extra columns (such as CLIENTNUM) are kept "
                   "out of the model. Open-to-buy and utilisation are computed from the limit and balance.")
            ui.footer()
            return
        try:
            raw = pd.read_csv(file)
        except Exception:  # noqa: BLE001 (anything pandas cannot parse)
            st.error("That file could not be read as a CSV.", icon=":material/error:")
            return
        rows, errors = validate_rows(raw)
        if errors:
            st.warning(f"{len(raw) - len(rows):,} of {len(raw):,} rows were skipped:\n\n- " + "\n- ".join(errors),
                       icon=":material/warning:")
        if rows.empty:
            return
        scored = _score(rows)
        if "CLIENTNUM" in raw:
            scored.insert(0, "CLIENTNUM", raw.loc[rows.index, "CLIENTNUM"].values)
    else:
        test = data.test_set()
        scored = _score(test.drop(columns=["row_id", "churned", "proba"]))
        scored.insert(0, "Customer", test["row_id"].values)
        outcome = test["churned"].to_numpy()
        scored.insert(4, "Actually churned", test["churned"].map({1: "Yes", 0: "No"}).values)

    proba = scored["Churn probability"].to_numpy()
    t = st.slider("Flag customers at or above this churn probability", 0.05, 0.95, float(m.threshold), 0.01,
                  format="%.2f", help=f"The model's recommended threshold is {m.threshold:.2f} (best F2 score).")
    flagged = proba >= t

    c = st.columns(4)
    c[0].metric("Customers scored", f"{len(proba):,}")
    c[1].metric("Flagged for a call", f"{flagged.sum():,}", f"{flagged.mean():.0%} of the list", delta_color="off", delta_arrow="off")
    c[2].metric("Expected churners among them", f"{proba[flagged].sum():,.0f}",
                help="Sum of the flagged customers' probabilities. The model is cautious in the middle of the range, so "
                     "treat this as a slight underestimate (see Model performance).")
    if outcome is not None:
        caught = int((flagged & (outcome == 1)).sum())
        c[3].metric("Churners actually caught", f"{caught:,} of {int(outcome.sum()):,}",
                    f"{caught / outcome.sum():.0%} recall, {caught / max(flagged.sum(), 1):.0%} precision",
                    delta_color="off", delta_arrow="off")
    else:
        c[3].metric("Expected churners missed", f"{proba[~flagged].sum():,.0f}",
                    help="Sum of the probabilities of customers below the threshold.")

    ui.spotlight(
        "Most customers are clearly safe; churners stand apart",
        "Distribution of churn probabilities. Everything right of the dashed line gets a call. "
        + ("Red bars are customers who really churned, blue those who stayed." if outcome is not None else ""),
        charts.risk_histogram(proba, t, outcome), key="hist",
    )

    st.subheader("Ranked call list", anchor=False)
    view = scored.sort_values("Churn probability", ascending=False)
    only = st.toggle("Show flagged customers only", value=True)
    if only:
        view = view[view["Churn probability"] >= t]
    show = view.rename(columns=LABELS)
    st.dataframe(
        show, hide_index=True, height=420,
        column_config={"Churn probability": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                            max_value=1)},
    )
    st.download_button("Download the scored list (CSV)", view.to_csv(index=False), "churn_scores.csv", "text/csv",
                       icon=":material/download:")
    ui.footer()
