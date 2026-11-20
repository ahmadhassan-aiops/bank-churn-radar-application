# Bank Churn Radar: credit card churn prediction application

**Live app:** [bank-churn-radar-application.streamlit.app](https://bank-churn-radar-application.streamlit.app/) · **Notebook and project folder:** [AIOPS-Portfolio-Projects](https://github.com/ahmadhassan-aiops/AIOPS-Portfolio-Projects/tree/main/2.%20Machine%20Learning%20Projects/Bank%20churners%20project)

A churn model for a bank's credit card customers, by **Ahmad Hassan** (AI Engineer), served as an interactive tool
for a retention team. A tuned XGBoost pipeline, trained on 8,101 customers and tested once on 2,026 it never saw,
**catches 94% of the customers who close their card, and 84% of the customers it flags really do churn**
(PR-AUC 0.968).

The app is a light **retention radar console**: customers are blips on a radar scope (closer to the centre = more
likely to churn, bearing = card transactions), lists become flight-strip style boards, and every block sits in its own
bracketed console module.

- **Track a customer**: enter a profile and 12 months of card activity; see the customer on the scope among the
  held-out customers, the churn probability with its Clear / Watch / Call state, the signals (SHAP values) that drive
  the score next to a typical stayer, and retention orders matched to those signals. Updates live.
- **Strip board**: score a CSV (or the held-out customers), set the call line, see the whole list on the scope, and
  work the call, watch and clear bays of customer strips; download the ranked list.
- **Calibration**: held-out results, the call line per 1,000 customers with a reached / missed / extra-call grid, the
  recall-precision trade-off, calibration and what drives the scores.
- **Manual**: the data, how the model was chosen, limits and code.

## Why this stack

| Need | Choice |
|---|---|
| Interactive scoring, what-if exploration and charts for business users | **Streamlit** (a Flask or Django front end would rebuild the same widgets by hand) |
| One artefact from raw record to probability, no training/serving skew | **scikit-learn `Pipeline`** (imputation + one-hot encoding + model), saved with `joblib` |
| Best ranking on tabular data, the best-calibrated probabilities of the finalists, fast exact explanations | **XGBoost** with native SHAP values (`pred_contribs`), no extra explainer library |
| Reject bad input before it reaches the model | **pydantic** schema for single customers and every row of an upload |
| Confidence that the shipped file is the evaluated model | **pytest** + GitHub Actions: predictions, test metrics, SHAP additivity, validation and every page |

The inference code (`churn/`) has no Streamlit dependency, so the same model can be served by an API
(FastAPI, for example) when a CRM needs to call it.

## How it is built

```
app.py                 entry point (console bar and status line, native navigation hidden)
nav.py                 page registry
views/                 one module per page
churn/schema.py        pydantic customer schema, labels, upload validation
churn/model.py         loads the pipeline, scores, explains (SHAP), risk bands
churn/actions.py       retention playbook: drivers -> actions
scope/                 console UI kit (modules, readouts, strips, signals), radar and other charts, theme, loaders, links
model/                 bank_churn_xgb.joblib + metadata.json (threshold, metrics, versions)
data/                  held-out test customers with scores, reference statistics
train.py               rebuilds model/ and data/ from the raw CSV
tests/                 model, schema and page tests
```

## Run locally

```bash
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

Rebuild the model from the dataset (same split, parameters and threshold as the notebook):

```bash
python train.py "path/to/BankChurners.csv"
```

## Data

The public Credit Card Customers dataset (10,127 anonymised customers; Thera Bank is the case-study name).
Predictions are a demonstration, not financial advice.
