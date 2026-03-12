from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import feature_engineer, model_input_columns


class ChurnPredictor:
    def __init__(self, model_path: Path, threshold: float = 0.5):
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.model = artifact["pipeline"]
            self.threshold = float(artifact.get("threshold", threshold))
        else:
            self.model = artifact
            self.threshold = threshold

    def predict_one(self, payload: dict) -> dict:
        customer_id = payload["customer_id"]
        frame = pd.DataFrame([{k: payload[k] for k in model_input_columns()}])
        transformed = feature_engineer(frame)

        probability = float(self.model.predict_proba(transformed)[:, 1][0])
        prediction = "yes" if probability >= self.threshold else "no"

        return {
            "customer_id": customer_id,
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "risk_band": self._risk_band(probability),
            "top_reasons": self._reason_codes(payload),
        }

    def predict_batch(self, payloads: list[dict]) -> list[dict]:
        return [self.predict_one(payload) for payload in payloads]

    @staticmethod
    def _risk_band(probability: float) -> str:
        if probability >= 0.7:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _reason_codes(payload: dict) -> list[str]:
        reasons = []
        if payload["support_tickets"] >= 4:
            reasons.append("high_support_volume")
        if payload["payment_history"] in {"failed", "chargeback", "delayed"}:
            reasons.append("payment_risk")
        if payload["login_activity"] < 10:
            reasons.append("low_login_activity")
        if payload["tenure_months"] < 6:
            reasons.append("new_customer_risk")
        if payload["usage_frequency"] < 8:
            reasons.append("low_usage")

        if not reasons:
            reasons.append("stable_behavior")
        return reasons[:3]


def load_metrics(metrics_path: Path) -> dict:
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}
