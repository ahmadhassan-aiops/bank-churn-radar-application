"""Manual: the dataset, how the model was chosen, the limits and how the app is built."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from scope import data, ui
from scope.site import CODE_URL, NOTEBOOK_URL, PROJECT_URL

# Cross-validated results of the tuned models (5-fold on the training set, threshold 0.5), from the notebook
TUNED = {
    "Model": ["XGBoost (chosen)", "Gradient boosting", "Random forest"],
    "Recall": ["0.891", "0.888", "0.838"],
    "Precision": ["0.935", "0.941", "0.932"],
    "F2": ["0.899", "0.898", "0.855"],
    "PR-AUC": ["0.972", "0.972", "0.952"],
    "Out-of-fold log loss": ["0.073", "0.096", "not a finalist"],
}


def render() -> None:
    m = data.model()
    meta = m.meta
    ui.page("Mode 4 · Manual", "How the model was built",
            "The dataset, the modelling choices and their reasons, and what the model cannot know.")
    ui.readouts([("Customers", "10,127", "public Credit Card Customers data"), ("Closed their card", "1,627", "16.1%"),
                 ("Candidate models", "25", "7 algorithms × 3 data treatments + class weights"),
                 ("Test set", f"{meta['test_rows']:,}", "locked away, used once")])

    with ui.module("Manual 01", "The data"):
        st.markdown(
            "The public **Credit Card Customers** dataset: 10,127 customers of a bank (called Thera Bank in the case study), "
            "with demographics, their relationship with the bank and 12 months of card activity. **1,627 (16.1%) closed "
            "their card.** Education, marital status and income are missing for some customers (income is recorded as "
            "`abc` for 1,112); these become an `Unknown` category rather than being guessed.")

    with ui.module("Manual 02", "How the model was chosen"):
        st.markdown(
            "1. **Split first.** 20% of customers (stratified) were locked away as the test set before any modelling.\n"
            "2. **Pre-processing inside a pipeline.** Missing categories become `Unknown` and categories are one-hot "
            "encoded, both fitted on training folds only, so nothing leaks from validation data. The same pipeline is what "
            "this app loads.\n"
            "3. **25 candidates.** Seven algorithms (logistic regression, decision tree, bagging, random forest, AdaBoost, "
            "gradient boosting, XGBoost) on original, SMOTE-oversampled and undersampled data, plus class weights, all with "
            "the same 5-fold cross-validation. Resampling happens inside each fold.\n"
            "4. **Metric.** PR-AUC (how well customers are ranked by risk) to compare and tune; F2, which weighs recall twice "
            "as much as precision, to set the call line. A missed churner costs more than an extra call.\n"
            "5. **Tuning.** The three strongest algorithms were tuned with 40 random combinations each. XGBoost and gradient "
            "boosting tied on PR-AUC, so the tie went to the better-calibrated model: XGBoost's probabilities are closer to "
            "reality (lower log loss), while tuned gradient boosting overfitted (perfect training scores, probabilities "
            "pushed to 0 and 1).\n"
            f"6. **Call line.** {m.threshold:.2f}, the best F2 on out-of-fold predictions.\n"
            "7. **One test.** The final model scored the test set once.")
        st.dataframe(pd.DataFrame(TUNED), hide_index=True, width="stretch")
        ui.cap("Tuned models, 5-fold cross-validation on the training set, at the default 0.5 threshold.")

    a, b = st.columns(2, gap="medium")
    with a, ui.module("Manual 03", "Limits"):
        st.markdown(
            "- **One snapshot.** Activity covers a single 12-month window with no dates, so the model cannot say *when* a "
            "customer will leave.\n"
            "- **Contacts may include the goodbye.** All 54 customers with 6 contacts churned. Some contacts may be the call "
            "to close the card, which would make this feature partly a symptom rather than a warning.\n"
            "- **Demographics barely matter** (churn varies by a few points across income, education and marital status), "
            "so the model leans on behaviour. That is good for fairness, but a customer whose behaviour has not changed yet "
            "will score low.\n"
            "- **Probabilities hold for this bank's mix** of customers. Another portfolio would need re-checking.\n"
            "- **Not financial advice.** A demonstration on a public dataset.")
    with b, ui.module("Manual 04", "Code and reproducibility"):
        st.markdown(
            f"- Notebook with the full analysis and model selection: [bank churners.ipynb]({NOTEBOOK_URL}) "
            f"([project folder]({PROJECT_URL}))\n"
            f"- App code, tests and `train.py`, which rebuilds the shipped model from the CSV: [GitHub]({CODE_URL})\n"
            f"- Built with scikit-learn {meta['versions']['scikit-learn']}, XGBoost {meta['versions']['xgboost']}, pandas, "
            "Plotly and Streamlit. The `churn` package (schema, model, actions) has no Streamlit code.")
    ui.footer()
