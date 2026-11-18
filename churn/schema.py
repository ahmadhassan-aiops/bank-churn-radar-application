"""The customer record the model scores, validated with pydantic.

Ranges are the ones seen in the training data (10,127 customers), so the app never asks the model to
extrapolate far outside what it learned. Open-to-buy and utilisation are not inputs: they are exactly
credit limit minus revolving balance, and balance divided by limit, so they are derived here.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

GENDERS = ("F", "M")
EDUCATION = ("Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown")
MARITAL = ("Single", "Married", "Divorced", "Unknown")
INCOME = ("Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown")
CARDS = ("Blue", "Silver", "Gold", "Platinum")

CATEGORICAL = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
NUMERICAL = [
    "Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
]
MODEL_COLUMNS = [  # the dataset's column order, which the pipeline was fitted with
    "Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category",
    "Card_Category", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
    "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
    "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
]
DERIVED = ("Avg_Open_To_Buy", "Avg_Utilization_Ratio")
INPUT_COLUMNS = [c for c in MODEL_COLUMNS if c not in DERIVED]

# Plain-language names, used by the form, the explanations and the tables
LABELS = {
    "Customer_Age": "Age",
    "Gender": "Gender",
    "Dependent_count": "Dependents",
    "Education_Level": "Education",
    "Marital_Status": "Marital status",
    "Income_Category": "Annual income",
    "Card_Category": "Card",
    "Months_on_book": "Months as a customer",
    "Total_Relationship_Count": "Products held with the bank",
    "Months_Inactive_12_mon": "Inactive months (last 12)",
    "Contacts_Count_12_mon": "Contacts with the bank (last 12 months)",
    "Credit_Limit": "Credit limit",
    "Total_Revolving_Bal": "Revolving balance",
    "Avg_Open_To_Buy": "Credit still available",
    "Total_Amt_Chng_Q4_Q1": "Spend change, Q4 to Q1",
    "Total_Trans_Amt": "Spend in the last 12 months",
    "Total_Trans_Ct": "Transactions in the last 12 months",
    "Total_Ct_Chng_Q4_Q1": "Transaction count change, Q4 to Q1",
    "Avg_Utilization_Ratio": "Credit utilisation",
}


class Customer(BaseModel):
    """One credit card customer, as the bank's systems describe them."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    Customer_Age: int = Field(ge=18, le=100)
    Gender: Literal[GENDERS]
    Dependent_count: int = Field(ge=0, le=10)
    Education_Level: Literal[EDUCATION] = "Unknown"
    Marital_Status: Literal[MARITAL] = "Unknown"
    Income_Category: Literal[INCOME] = "Unknown"
    Card_Category: Literal[CARDS]
    Months_on_book: int = Field(ge=0, le=120)
    Total_Relationship_Count: int = Field(ge=1, le=6)
    Months_Inactive_12_mon: int = Field(ge=0, le=12)
    Contacts_Count_12_mon: int = Field(ge=0, le=12)
    Credit_Limit: float = Field(gt=0, le=100_000)
    Total_Revolving_Bal: float = Field(ge=0, le=100_000)
    Total_Amt_Chng_Q4_Q1: float = Field(ge=0, le=10)
    Total_Trans_Amt: float = Field(ge=0, le=100_000)
    Total_Trans_Ct: int = Field(ge=0, le=500)
    Total_Ct_Chng_Q4_Q1: float = Field(ge=0, le=10)

    @model_validator(mode="after")
    def _balance_within_limit(self) -> "Customer":
        if self.Total_Revolving_Bal > self.Credit_Limit:
            raise ValueError("Revolving balance cannot be larger than the credit limit.")
        return self

    @property
    def open_to_buy(self) -> float:
        return self.Credit_Limit - self.Total_Revolving_Bal

    @property
    def utilisation(self) -> float:
        return round(self.Total_Revolving_Bal / self.Credit_Limit, 3)

    def to_frame(self) -> pd.DataFrame:
        return to_model_frame(pd.DataFrame([self.model_dump()]))


def _clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values and the dataset's 'abc' placeholder become 'Unknown', as in training."""
    for col in ("Education_Level", "Marital_Status", "Income_Category"):
        df[col] = df[col].replace("abc", None).fillna("Unknown").astype(str)
    return df


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows -> the exact columns the pipeline expects (derived fields filled in)."""
    df = df.copy()
    if "Avg_Open_To_Buy" not in df:
        df["Avg_Open_To_Buy"] = df["Credit_Limit"] - df["Total_Revolving_Bal"]
    if "Avg_Utilization_Ratio" not in df:
        df["Avg_Utilization_Ratio"] = (df["Total_Revolving_Bal"] / df["Credit_Limit"]).round(3)
    return _clean_categories(df)[MODEL_COLUMNS]


def validate_rows(df: pd.DataFrame, max_errors: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """Check an uploaded table row by row. Returns the valid rows (model columns) and readable errors."""
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        return df.iloc[0:0], [f"Missing columns: {', '.join(missing)}"]
    df = _clean_categories(df.copy())
    keep, errors = [], []
    for i, row in enumerate(df[INPUT_COLUMNS].to_dict("records")):
        try:
            Customer(**row)
            keep.append(i)
        except ValidationError as exc:
            if len(errors) < max_errors:
                err = exc.errors()[0]
                field = err["loc"][0] if err["loc"] else ""
                where = f"{LABELS.get(field, field)}: " if field else ""
                errors.append(f"Row {i + 2}: {where}{err['msg'].removeprefix('Value error, ')}")
    valid = df.iloc[keep]
    return to_model_frame(valid), errors
