from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "usage_frequency",
    "login_activity",
    "support_tickets",
    "avg_session_minutes",
    "monthly_spend",
    "tenure_months",
    "ticket_per_login",
    "engagement_score",
    "payment_risk_score",
]

CATEGORICAL_FEATURES = [
    "subscription_type",
    "payment_history",
    "region",
    "high_support_flag",
    "low_tenure_flag",
]

TARGET_COLUMN = "churn"


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    safe_login = data["login_activity"].fillna(0) + 1
    safe_usage = data["usage_frequency"].fillna(0)
    safe_tickets = data["support_tickets"].fillna(0)

    data["ticket_per_login"] = safe_tickets / safe_login
    data["engagement_score"] = 0.6 * safe_login + 0.4 * safe_usage

    payment_map = {
        "on_time": 0.0,
        "delayed": 1.0,
        "failed": 2.0,
        "chargeback": 3.0,
    }
    data["payment_risk_score"] = data["payment_history"].map(payment_map).fillna(1.0)

    data["high_support_flag"] = np.where(safe_tickets >= 4, "yes", "no")
    data["low_tenure_flag"] = np.where(data["tenure_months"].fillna(0) < 6, "yes", "no")

    return data


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = feature_engineer(df.drop(columns=[TARGET_COLUMN]))
    target = df[TARGET_COLUMN].astype(int)
    return features, target


def model_input_columns() -> Iterable[str]:
    return [
        "usage_frequency",
        "subscription_type",
        "login_activity",
        "support_tickets",
        "payment_history",
        "avg_session_minutes",
        "monthly_spend",
        "tenure_months",
        "region",
    ]
