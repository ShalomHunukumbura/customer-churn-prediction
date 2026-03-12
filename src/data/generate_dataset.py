from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, RANDOM_STATE
from src.utils import ensure_dir


def _inject_missing(df: pd.DataFrame, columns: list[str], rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


def build_synthetic_dataset(n_rows: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    subscription_type = rng.choice(["monthly", "quarterly", "annual"], size=n_rows, p=[0.58, 0.24, 0.18])
    payment_history = rng.choice(["on_time", "delayed", "failed", "chargeback"], size=n_rows, p=[0.67, 0.2, 0.1, 0.03])
    region = rng.choice(["NA", "EU", "APAC", "LATAM", "MEA"], size=n_rows, p=[0.35, 0.2, 0.25, 0.12, 0.08])

    usage_frequency = np.clip(rng.normal(16, 7, n_rows), 1, 45)
    login_activity = np.clip(rng.normal(22, 11, n_rows), 0, 80)
    support_tickets = np.clip(rng.poisson(1.6, n_rows), 0, 12)
    avg_session_minutes = np.clip(rng.normal(15, 6, n_rows), 2, 60)
    monthly_spend = np.clip(rng.normal(84, 28, n_rows), 10, 260)
    tenure_months = np.clip(rng.gamma(shape=2.1, scale=7.5, size=n_rows), 1, 72)

    annual_discount = np.where(subscription_type == "annual", 0.9, 1.0)
    monthly_spend = monthly_spend * annual_discount

    low_engagement = ((usage_frequency < 9) & (login_activity < 12)).astype(float)
    payment_risk = (
        1.2 * (payment_history == "failed").astype(float)
        + 1.45 * (payment_history == "chargeback").astype(float)
        + 0.5 * (payment_history == "delayed").astype(float)
    )
    tenure_risk = np.where(tenure_months < 6, 0.85, np.where(tenure_months < 12, 0.35, -0.2))
    plan_risk = np.where(subscription_type == "monthly", 0.55, np.where(subscription_type == "quarterly", 0.18, -0.24))
    support_risk = 0.22 * np.clip(support_tickets - 1, 0, None)
    region_risk = np.where(np.isin(region, ["LATAM", "MEA"]), 0.12, 0.0)

    score = (
        -1.35
        + payment_risk
        + support_risk
        + plan_risk
        + tenure_risk
        + region_risk
        + 0.7 * low_engagement
        - 0.012 * login_activity
        - 0.016 * usage_frequency
        - 0.01 * avg_session_minutes
        + rng.normal(0, 0.35, n_rows)
    )

    probability = 1 / (1 + np.exp(-score))
    churn = (rng.random(n_rows) < probability).astype(int)

    dataset = pd.DataFrame(
        {
            "usage_frequency": usage_frequency.round(2),
            "subscription_type": subscription_type,
            "login_activity": login_activity.round(2),
            "support_tickets": support_tickets.astype(int),
            "payment_history": payment_history,
            "avg_session_minutes": avg_session_minutes.round(2),
            "monthly_spend": monthly_spend.round(2),
            "tenure_months": tenure_months.round(0).astype(int),
            "region": region,
            "churn": churn,
        }
    )

    return _inject_missing(
        dataset,
        columns=[
            "usage_frequency",
            "login_activity",
            "payment_history",
            "avg_session_minutes",
            "region",
        ],
        rate=0.06,
        seed=seed + 10,
    )


def save_dataset(df: pd.DataFrame, path: Path = RAW_DATA_PATH) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    frame = build_synthetic_dataset()
    save_dataset(frame)
    churn_rate = float(frame["churn"].mean())
    print(f"Saved synthetic dataset with {len(frame)} rows to {RAW_DATA_PATH}")
    print(f"Churn rate: {churn_rate:.3f}")
