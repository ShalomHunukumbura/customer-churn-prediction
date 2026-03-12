from __future__ import annotations

def test_health_endpoint_returns_known_status() -> None:
    from src.api import health

    result = health()
    assert result["status"] in {"ok", "model_missing"}


def test_predict_batch_validation_model() -> None:
    from src.schemas import BatchChurnRequest, ChurnRequest

    payload = BatchChurnRequest(
        records=[
            ChurnRequest(
                customer_id="C-1",
                usage_frequency=10,
                subscription_type="monthly",
                login_activity=12,
                support_tickets=2,
                payment_history="on_time",
                avg_session_minutes=20,
                monthly_spend=88,
                tenure_months=7,
                region="NA",
            )
        ]
    )
    assert len(payload.records) == 1
