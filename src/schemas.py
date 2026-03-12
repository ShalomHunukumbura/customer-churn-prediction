from __future__ import annotations

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    usage_frequency: float = Field(..., ge=0)
    subscription_type: str
    login_activity: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    payment_history: str
    avg_session_minutes: float = Field(..., ge=0)
    monthly_spend: float = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0)
    region: str


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: str
    risk_band: str
    top_reasons: list[str]


class BatchChurnRequest(BaseModel):
    records: list[ChurnRequest] = Field(..., min_length=1, max_length=500)


class BatchChurnResponse(BaseModel):
    predictions: list[ChurnResponse]
