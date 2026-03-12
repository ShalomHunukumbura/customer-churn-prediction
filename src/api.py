from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import ChurnPredictor, load_metrics
from src.schemas import BatchChurnRequest, BatchChurnResponse, ChurnRequest, ChurnResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_predictor()
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

predictor: ChurnPredictor | None = None


def _load_predictor() -> None:
    global predictor
    if predictor is None and MODEL_PATH.exists():
        predictor = ChurnPredictor(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    status = "ok" if MODEL_PATH.exists() else "model_missing"
    return {"status": status}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    _load_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact not available")
    return ChurnResponse(**predictor.predict_one(payload.model_dump()))


@app.post("/predict/batch", response_model=BatchChurnResponse)
def predict_batch(payload: BatchChurnRequest) -> BatchChurnResponse:
    _load_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact not available")
    rows = [record.model_dump() for record in payload.records]
    return BatchChurnResponse(predictions=[ChurnResponse(**row) for row in predictor.predict_batch(rows)])


@app.get("/model/info")
def model_info() -> dict:
    return {
        "model_path": str(MODEL_PATH),
        "metrics": load_metrics(METRICS_PATH),
    }
