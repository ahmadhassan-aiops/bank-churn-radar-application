"""Turn the features that push a customer's risk up into retention actions a relationship manager can take."""

from __future__ import annotations

import pandas as pd

# feature -> (Material icon, action title, why and what to do)
PLAYBOOK: dict[str, tuple[str, str, str]] = {
    "Total_Trans_Ct": (
        "shopping_bag", "Win back everyday spending",
        "Few transactions is the strongest churn signal. Offer cashback or bonus points on the next purchases "
        "(groceries, fuel, subscriptions) to make this the card they reach for.",
    ),
    "Total_Trans_Amt": (
        "payments", "Grow spend on the card",
        "Annual spend is low for this customer. A spend-and-get offer or a category bonus gives them a reason "
        "to move spending back to this card.",
    ),
    "Total_Ct_Chng_Q4_Q1": (
        "trending_down", "Act now: activity is falling",
        "Transactions dropped from Q4 to Q1, a sign the customer is already moving to another card. "
        "A personal call this month beats a campaign next quarter.",
    ),
    "Total_Amt_Chng_Q4_Q1": (
        "trending_down", "Spending is sliding",
        "Spend fell from Q4 to Q1. Check whether a competitor offer, a fee or a declined transaction "
        "triggered it.",
    ),
    "Total_Revolving_Bal": (
        "credit_card_off", "Make the card worth keeping",
        "No or little revolving balance: the card is not their main credit line. A balance-transfer rate or "
        "a fee waiver keeps it useful.",
    ),
    "Avg_Utilization_Ratio": (
        "credit_card_off", "Low use of the credit line",
        "The customer uses little of their limit. Promote features they are not using (instalments, "
        "contactless, rewards).",
    ),
    "Total_Relationship_Count": (
        "account_balance", "Deepen the relationship",
        "Customers with one or two products churn at about 27%, against 11% with four or more. Offer a second "
        "product: a savings account, insurance or a bundle.",
    ),
    "Contacts_Count_12_mon": (
        "support_agent", "Resolve open issues",
        "Many contacts with the bank in a year often mean unresolved problems. Have a senior agent review the "
        "contact history and close the loop.",
    ),
    "Months_Inactive_12_mon": (
        "notifications_active", "Re-activate the card",
        "Several inactive months. A re-activation offer or a reminder of the card's benefits brings dormant "
        "customers back.",
    ),
    "Credit_Limit": (
        "tune", "Review the credit limit",
        "The limit may not suit how the customer uses the card. A limit review signals that the bank values them.",
    ),
    "Avg_Open_To_Buy": (
        "tune", "Review the credit limit",
        "The available credit pattern adds to the risk. A limit review signals that the bank values them.",
    ),
    "Months_on_book": (
        "loyalty", "Recognise their tenure",
        "Tenure adds to the risk for this profile. A loyalty thank-you or anniversary reward costs little.",
    ),
}

FALLBACK = (
    "handshake", "Keep the relationship warm",
    "No single behaviour stands out. Keep the customer in regular communications and watch the next quarter's "
    "activity.",
)


HIGH_BALANCE = (
    "credit_score", "Ease a heavy balance",
    "For this customer a balance near the top of the range adds to the risk: they may be under strain or "
    "shopping for a cheaper rate. Offer an instalment plan or a lower rate on the balance.",
)


# Each playbook text describes a situation ("spend is low", "many contacts"). It is only shown when the customer's
# value really is in that situation; otherwise the feature raises the score through an interaction with other
# values, and the text would be wrong for this customer. Thresholds are around the typical (median) stayer.
APPLIES = {
    "Total_Trans_Ct": lambda v: v < 71,
    "Total_Trans_Amt": lambda v: v < 4100,
    "Total_Ct_Chng_Q4_Q1": lambda v: v < 1,
    "Total_Amt_Chng_Q4_Q1": lambda v: v < 1,
    "Total_Revolving_Bal": lambda v: v < 500,
    "Avg_Utilization_Ratio": lambda v: v < 0.1,
    "Total_Relationship_Count": lambda v: v <= 3,
    "Contacts_Count_12_mon": lambda v: v >= 3,
    "Months_Inactive_12_mon": lambda v: v >= 3,
}


def _action(feature: str, customer: pd.Series | None) -> tuple[str, str, str] | None:
    """The playbook entry if it fits this customer's actual value (always, when no customer is given)."""
    if customer is None:
        return PLAYBOOK.get(feature)
    if feature in ("Total_Revolving_Bal", "Avg_Utilization_Ratio") and customer.get("Total_Revolving_Bal", 0) >= 2000:
        return HIGH_BALANCE
    check = APPLIES.get(feature)
    if check is not None and feature in customer and not check(customer[feature]):
        return None
    return PLAYBOOK.get(feature)


def plan(contributions: pd.Series, customer: pd.Series | None = None,
         limit: int = 3) -> list[tuple[str | None, tuple[str, str, str]]]:
    """(feature, action) pairs for the features that raise the churn score most, one per distinct action.

    `customer` (the model row) lets an action depend on the value, e.g. a zero balance versus one at the cap.
    The feature is None for the fallback action.
    """
    out, seen = [], set()
    for feature, value in contributions.sort_values(ascending=False).items():
        if value <= 0.25 or len(out) == limit:  # ignore features that barely raise the score
            break
        action = _action(feature, customer)
        if action and action[1] not in seen:
            seen.add(action[1])
            out.append((feature, action))
    return out or [(None, FALLBACK)]


def suggest(contributions: pd.Series, customer: pd.Series | None = None, limit: int = 3) -> list[tuple[str, str, str]]:
    """The actions alone (see plan)."""
    return [action for _, action in plan(contributions, customer, limit)]
