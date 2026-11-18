"""Cached loaders: the model, the held-out test customers and the reference statistics (built by train.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from churn.model import ChurnModel

DATA = Path(__file__).resolve().parent.parent / "data"


@st.cache_resource(show_spinner="Loading the model...")
def model() -> ChurnModel:
    return ChurnModel()


@st.cache_data
def test_set() -> pd.DataFrame:
    """The 2,026 customers the model never saw in training, with their outcome and churn probability."""
    return pd.read_parquet(DATA / "test_scored.parquet")


@st.cache_data
def reference() -> dict:
    """Medians for customers who stayed and churned, feature importance and the threshold curve."""
    return json.loads((DATA / "reference.json").read_text(encoding="utf-8"))
