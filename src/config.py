from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "churn_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "churn_model.pkl"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", MODELS_DIR / "metrics.json"))
