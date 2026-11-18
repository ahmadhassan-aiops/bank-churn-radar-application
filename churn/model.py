"""Load the trained pipeline and score customers. No Streamlit here, so any front end (or an API) can use it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from churn.schema import CATEGORICAL, MODEL_COLUMNS, Customer, to_model_frame

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


@dataclass(frozen=True)
class Band:
    name: str
    lower: float  # inclusive lower bound on the churn probability


class ChurnModel:
    """The saved scikit-learn pipeline (imputation, one-hot encoding, XGBoost) plus its metadata."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pipeline = joblib.load(model_dir / "bank_churn_xgb.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.threshold: float = self.meta["threshold"]
        # Low below half the threshold; Watch up to the threshold; Flag (call the customer) from the threshold up
        self.bands = (Band("Flag", self.threshold), Band("Watch", self.threshold / 2), Band("Low", 0.0))

    # -- scoring -----------------------------------------------------------------------------------
    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Churn probability for rows that already have the model columns (see schema.to_model_frame)."""
        return self.pipeline.predict_proba(rows[MODEL_COLUMNS])[:, 1]

    def score(self, customer: Customer) -> float:
        return float(self.predict_proba(customer.to_frame())[0])

    def score_frame(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(to_model_frame(raw))

    def band(self, p: float) -> str:
        return next(b.name for b in self.bands if p >= b.lower)

    # -- explanations ------------------------------------------------------------------------------
    @cached_property
    def _source(self) -> list[str]:
        """For each encoded column, the original feature it came from (one-hot columns -> their category)."""
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        return [next((c for c in CATEGORICAL if n.startswith(c + "_")), n) for n in names]

    def explain(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """SHAP contributions per original feature (log-odds) and the bias, computed natively by XGBoost.

        For each row, bias + sum(contributions) is the model's log-odds of churn.
        """
        encoded = self.pipeline.named_steps["prep"].transform(rows[MODEL_COLUMNS])
        booster = self.pipeline.named_steps["model"].get_booster()
        contribs = booster.predict(xgb.DMatrix(encoded), pred_contribs=True)
        per_column = pd.DataFrame(contribs[:, :-1], index=rows.index)
        grouped = per_column.T.groupby(np.array(self._source)).sum().T
        return grouped[MODEL_COLUMNS], contribs[:, -1]


def to_logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def to_prob(logit: float) -> float:
    return float(1 / (1 + np.exp(-logit)))
