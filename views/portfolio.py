"""Strip board: score a whole customer list, see it on the scope, sort it into call / watch / clear bays of strips,
and download the ranked list."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.schema import INPUT_COLUMNS, LABELS, validate_rows
from scope import charts, data, ui

TEST = "The 2,026 held-out customers (outcomes known)"
UPLOAD = "Upload my own CSV"
SHOWN = 6  # strips per bay


@st.cache_data(show_spinner="Scoring customers...")
def _score(rows: pd.DataFrame) -> pd.DataFrame:
    m = data.model()
    proba = m.predict_proba(rows)
    contrib, _ = m.explain(rows)
    top = contrib.idxmax(axis=1).where(contrib.max(axis=1) > 0)
    out = rows.copy()
    out.insert(0, "Churn probability", proba)
    out.insert(1, "Main reason", top.map(LABELS).fillna("Nothing raises the risk"))
    return out


def _band(p: np.ndarray, t: float) -> np.ndarray:
    return np.where(p >= t, "Flag", np.where(p >= t / 2, "Watch", "Low"))


def render() -> None:
    m = data.model()
    ui.page("Mode 2 · Strip board", "Who should the retention team call first?",
            "Score a whole customer list at once. Every customer becomes a strip in the <b>call</b>, <b>watch</b> or "
            "<b>clear</b> bay, ranked by risk, and the scope shows how many calls it takes to reach how many of the "
            "customers who would leave.")

    with ui.module("Input 01", "Customer list", "The held-out customers (with what really happened), or your own CSV."):
        source = st.segmented_control("Customers to score", [TEST, UPLOAD], default=TEST, required=True,
                                      label_visibility="collapsed")
        outcome = None
        if source == UPLOAD:
            template = data.test_set()[INPUT_COLUMNS].head(5)
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            file = c1.file_uploader("CSV with one row per customer, using the dataset's column names", type="csv")
            c2.download_button("Template (5 rows)", template.to_csv(index=False), "churn_template.csv", "text/csv",
                               icon=":material/download:", width="stretch")
            if file is None:
                ui.cap("Required columns: " + ", ".join(INPUT_COLUMNS) + ". Extra columns (such as CLIENTNUM) are kept out "
                       "of the model. Open-to-buy and utilisation are computed from the limit and balance.")
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
            ids = raw.loc[rows.index, "CLIENTNUM"].astype(str).values if "CLIENTNUM" in raw else (rows.index + 2).astype(str)
            scored.insert(0, "Customer", ids)
        else:
            test = data.test_set()
            scored = _score(test.drop(columns=["row_id", "churned", "proba"]))
            scored.insert(0, "Customer", test["row_id"].astype(str).values)
            outcome = test["churned"].to_numpy()
            scored.insert(3, "Actually churned", test["churned"].map({1: "Yes", 0: "No"}).values)
        t = st.slider("Call line: call customers at or above this churn probability", 0.05, 0.95, float(m.threshold), 0.01,
                      format="%.2f", help=f"The model's recommended call line is {m.threshold:.2f} (best F2 score in cross-validation).")

    proba = scored["Churn probability"].to_numpy()
    band = _band(proba, t)
    flagged = band == "Flag"
    tiles = [("Customers scored", f"{len(proba):,}", "on the list"),
             ("Calls", f"{flagged.sum():,}", f"{flagged.mean():.0%} of the list", "#c0392b"),
             ("Expected churners called", f"{proba[flagged].sum():,.0f}", "sum of their probabilities")]
    if outcome is not None:
        caught = int((flagged & (outcome == 1)).sum())
        tiles.append(("Churners actually reached", f"{caught:,} / {int(outcome.sum()):,}",
                      f"{caught / outcome.sum():.0%} recall · {caught / max(flagged.sum(), 1):.0%} precision", "#0e7c66"))
    else:
        tiles.append(("Expected churners missed", f"{proba[~flagged].sum():,.0f}", "below the call line"))
    ui.readouts(tiles)

    order = np.argsort(-proba)

    def strips(name: str) -> list[str]:
        out = []
        for i in [i for i in order if band[i] == name][:SHOWN]:
            r = scored.iloc[i]
            tail = ("churned" if outcome[i] else "stayed") if outcome is not None else ""
            out.append(ui.strip(f"#{r['Customer']}", f"{r['Card_Category']} · {int(r['Months_on_book'])} mo", r["Main reason"],
                                f"{int(r['Total_Trans_Ct'])} txns · {int(r['Total_Relationship_Count'])} products · "
                                f"{int(r['Months_Inactive_12_mon'])} mo inactive", float(proba[i]), name, tail))
        return out

    def show(name: str) -> None:
        n = int((band == name).sum())
        ui.bay(name, n, strips(name), min(SHOWN, n))

    a, b = st.columns([1.05, 1], gap="medium")
    with a, ui.module("Scope 02", "The whole list on the radar",
                      "Closer to the centre = likelier to churn; the red ring is the call line you set. Bearing is the "
                      "number of transactions in the last 12 months."
                      + (" Red dots really churned." if outcome is not None else "")):
        st.plotly_chart(charts.radar(proba, scored["Total_Trans_Ct"].to_numpy(), t, outcome=outcome,
                                     ids=scored["Customer"].to_numpy(), height=440), config=charts.CONFIG, key="board_scope")
    with b, ui.module("Board 03", "Call bay", f"The {SHOWN} riskiest customers to call first. Each strip: customer, card and "
                      "tenure, the signal that raises their risk most, card activity and risk"
                      + (", and what really happened." if outcome is not None else ".")):
        show("Flag")
    with ui.module("Board 04", "Watch and clear bays", "Customers below the call line, riskiest first."):
        a, b = st.columns(2, gap="medium")
        with a:
            show("Watch")
        with b:
            show("Low")

    with ui.module("List 05", "Ranked call list", "Sort any column; the download holds every customer on the list."):
        view = scored.sort_values("Churn probability", ascending=False)
        view.insert(1, "Bay", pd.Series(_band(view["Churn probability"].to_numpy(), t), index=view.index)
                    .map({"Flag": "Call", "Watch": "Watch", "Low": "Clear"}))
        if st.toggle("Calls only", value=True, key="calls_only"):
            view = view[view["Churn probability"] >= t]
        st.dataframe(view.rename(columns=LABELS), hide_index=True, height=400, column_config={
            "Churn probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.download_button("Download the scored list (CSV)", view.to_csv(index=False), "churn_scores.csv", "text/csv",
                           icon=":material/download:")
    ui.footer()
