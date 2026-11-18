"""
Reproduce the model in model/ and the app's data/ files from the raw dataset.

The research (25 candidates, tuning, threshold choice) is in the notebook; this script replays the final
choice so the deployed artefact can be rebuilt and checked without a notebook:

    python train.py "path/to/BankChurners.csv"

It writes model/bank_churn_xgb.joblib, model/metadata.json, data/test_scored.parquet and data/reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from churn.model import ChurnModel
from churn.schema import CATEGORICAL, MODEL_COLUMNS, NUMERICAL

ROOT = Path(__file__).resolve().parent
PARAMS = {  # the notebook's tuned XGBoost (RandomizedSearchCV, 40 combinations, 5-fold PR-AUC)
    "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 2, "subsample": 0.9,
    "colsample_bytree": 0.5, "gamma": 0, "reg_lambda": 3,
}


def load(csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv)
    y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    X = df[MODEL_COLUMNS].copy()
    X["Income_Category"] = X["Income_Category"].replace("abc", np.nan)
    return X, y


def build() -> Pipeline:
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", "passthrough", NUMERICAL)],
                             verbose_feature_names_out=False)
    model = XGBClassifier(random_state=1, n_jobs=2, eval_metric="logloss", **PARAMS)
    return Pipeline([("prep", prep), ("model", model)])


def metrics(y, proba, t) -> dict:
    pred = (proba >= t).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred), "Recall": recall_score(y, pred), "Precision": precision_score(y, pred),
        "F1": f1_score(y, pred), "F2": fbeta_score(y, pred, beta=2), "PR-AUC": average_precision_score(y, proba),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def main(csv: Path) -> None:
    X, y = load(csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    pipe = build()
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    thresholds = np.round(np.arange(0.01, 0.951, 0.01), 2)
    threshold = float(thresholds[np.argmax([fbeta_score(y_train, oof >= t, beta=2) for t in thresholds])])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    test = {k: round(float(v), 4) for k, v in metrics(y_test, proba, threshold).items()}

    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "model" / "bank_churn_xgb.joblib")
    meta = {
        "model": "XGBoost",
        "threshold": threshold,
        "features": {"categorical": CATEGORICAL, "numerical": NUMERICAL},
        "params": PARAMS,
        "cv_pr_auc": round(float(average_precision_score(y_train, oof)), 4),
        "test": test,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "churn_rate": round(float(y.mean()), 4),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgb.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
    }
    (ROOT / "model" / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # App data: the test customers with outcome and probability, and reference statistics
    (ROOT / "data").mkdir(exist_ok=True)
    scored = X_test.copy()
    for col in ("Education_Level", "Marital_Status", "Income_Category"):
        scored[col] = scored[col].fillna("Unknown")
    scored.insert(0, "row_id", np.arange(1, len(scored) + 1))
    scored["churned"] = y_test.to_numpy()
    scored["proba"] = proba
    scored.reset_index(drop=True).to_parquet(ROOT / "data" / "test_scored.parquet", index=False)

    model = ChurnModel(ROOT / "model")
    contrib, _ = model.explain(X_test)
    train = X_train.assign(churned=y_train)
    reference = {
        "test": test,
        "median_stayed": train[train.churned == 0][NUMERICAL].median().round(3).to_dict(),
        "median_churned": train[train.churned == 1][NUMERICAL].median().round(3).to_dict(),
        "importance": {k: round(float(v), 3) for k, v in contrib.abs().mean().sort_values(ascending=False).items()},
    }
    (ROOT / "data" / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "BankChurners.csv")
