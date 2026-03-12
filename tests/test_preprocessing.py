from __future__ import annotations

import pandas as pd

from src.data_preprocessing import feature_engineer


def test_feature_engineering_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "usage_frequency": [10.0],
            "subscription_type": ["monthly"],
            "login_activity": [5.0],
            "support_tickets": [3],
            "payment_history": ["delayed"],
            "avg_session_minutes": [12.0],
            "monthly_spend": [90.0],
            "tenure_months": [4],
            "region": ["NA"],
        }
    )

    transformed = feature_engineer(frame)

    assert "ticket_per_login" in transformed.columns
    assert "engagement_score" in transformed.columns
    assert "payment_risk_score" in transformed.columns
    assert "high_support_flag" in transformed.columns
    assert "low_tenure_flag" in transformed.columns
