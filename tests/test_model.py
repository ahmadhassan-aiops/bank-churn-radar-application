"""Checks that the shipped model, its metadata and the inference code agree. Run: pytest -q"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score

from churn import actions
from churn.model import ChurnModel, to_logit
from churn.schema import INPUT_COLUMNS, MODEL_COLUMNS, Customer, validate_rows
from lib.data import DATA

TYPICAL = dict(
    Customer_Age=46, Gender="F", Dependent_count=2, Education_Level="Graduate", Marital_Status="Married",
    Income_Category="Less than $40K", Card_Category="Blue", Months_on_book=36, Total_Relationship_Count=4,
    Months_Inactive_12_mon=2, Contacts_Count_12_mon=2, Credit_Limit=4550, Total_Revolving_Bal=1280,
    Total_Amt_Chng_Q4_Q1=0.74, Total_Trans_Amt=3900, Total_Trans_Ct=67, Total_Ct_Chng_Q4_Q1=0.70,
)


@pytest.fixture(scope="module")
def model() -> ChurnModel:
    return ChurnModel()


@pytest.fixture(scope="module")
def test_set() -> pd.DataFrame:
    return pd.read_parquet(DATA / "test_scored.parquet")


def test_metadata_is_sane(model):
    assert 0 < model.threshold < 1
    assert model.meta["test"]["Recall"] > 0.85 and model.meta["test"]["Precision"] > 0.8


def test_shipped_predictions_reproduce(model, test_set):
    proba = model.predict_proba(test_set[MODEL_COLUMNS])
    assert np.allclose(proba, test_set["proba"], atol=1e-6)


def test_test_metrics_match_metadata(model, test_set):
    y, p = test_set["churned"], test_set["proba"]
    t = model.threshold
    assert recall_score(y, p >= t) == pytest.approx(model.meta["test"]["Recall"], abs=1e-4)
    assert precision_score(y, p >= t) == pytest.approx(model.meta["test"]["Precision"], abs=1e-4)
    assert average_precision_score(y, p) == pytest.approx(model.meta["test"]["PR-AUC"], abs=1e-4)


def test_shap_contributions_add_up(model, test_set):
    rows = test_set[MODEL_COLUMNS].head(50)
    contrib, bias = model.explain(rows)
    logits = np.array([to_logit(p) for p in model.predict_proba(rows)])
    assert np.allclose(contrib.sum(axis=1).to_numpy() + bias, logits, atol=1e-3)


def test_single_customer_scores_and_derives_fields(model):
    c = Customer(**TYPICAL)
    assert c.open_to_buy == 3270 and c.utilisation == pytest.approx(0.281)
    p = model.score(c)
    assert 0 <= p < model.threshold  # the median customer is not flagged


def test_activity_drop_raises_risk(model):
    quiet = Customer(**{**TYPICAL, "Total_Trans_Ct": 35, "Total_Trans_Amt": 1800, "Total_Revolving_Bal": 0,
                        "Contacts_Count_12_mon": 4, "Total_Ct_Chng_Q4_Q1": 0.4})
    assert model.score(quiet) > model.score(Customer(**TYPICAL))


def test_schema_rejects_bad_input():
    with pytest.raises(ValueError):
        Customer(**{**TYPICAL, "Total_Revolving_Bal": 9000})  # balance above the limit
    with pytest.raises(ValueError):
        Customer(**{**TYPICAL, "Card_Category": "Black"})


def test_validate_rows_keeps_good_rows_and_reports_bad(test_set):
    raw = test_set[INPUT_COLUMNS].head(5).copy()
    raw.loc[raw.index[1], "Card_Category"] = "Black"
    raw.loc[raw.index[3], "Income_Category"] = "abc"  # the dataset's placeholder: treated as Unknown
    rows, errors = validate_rows(raw)
    assert len(rows) == 4 and len(errors) == 1 and "Row 3" in errors[0]
    assert list(rows.columns) == MODEL_COLUMNS

    rows, errors = validate_rows(raw.drop(columns=["Credit_Limit"]))
    assert rows.empty and "Credit_Limit" in errors[0]


def test_actions_follow_the_drivers():
    contrib = pd.Series({"Total_Trans_Ct": 1.2, "Contacts_Count_12_mon": 0.4, "Customer_Age": -0.3})
    titles = [a[1] for a in actions.suggest(contrib)]
    assert titles[0] == "Win back everyday spending" and "Resolve open issues" in titles
    assert actions.suggest(pd.Series({"Customer_Age": -1.0}))[0] == actions.FALLBACK
    balance = pd.Series({"Total_Revolving_Bal": 0.8})
    assert actions.suggest(balance, pd.Series({"Total_Revolving_Bal": 0}))[0][1] == "Make the card worth keeping"
    assert actions.suggest(balance, pd.Series({"Total_Revolving_Bal": 2517}))[0] == actions.HIGH_BALANCE
