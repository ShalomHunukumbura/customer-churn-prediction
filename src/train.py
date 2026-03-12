from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODELS_DIR, RAW_DATA_PATH, RANDOM_STATE, REPORTS_DIR
from src.data_preprocessing import build_preprocessor, split_xy
from src.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_model_comparison,
    save_roc_pr_curves,
    save_report_markdown,
)
from src.utils import ensure_dir


def _best_threshold(y_true, y_proba) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.25, 0.81, 0.01):
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, float(best_f1)


def train_and_select(df: pd.DataFrame) -> dict:
    X, y = split_xy(df)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1200, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=RANDOM_STATE,
        ),
    }

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    results = {}
    fitted = {}
    thresholds = {}

    for name, model in models.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        threshold, tuned_f1 = _best_threshold(y_val, val_proba)
        val_pred = (val_proba >= threshold).astype(int)

        results[name] = compute_metrics(y_val, val_pred, val_proba)
        results[name]["threshold"] = threshold
        results[name]["tuned_f1"] = round(tuned_f1, 4)
        fitted[name] = pipe
        thresholds[name] = threshold

    final_name = max(results.keys(), key=lambda m: (results[m]["f1"], results[m]["pr_auc"]))
    final_model = fitted[final_name]
    final_threshold = thresholds[final_name]

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= final_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    test_metrics["threshold"] = final_threshold

    return {
        "model_name": final_name,
        "model": final_model,
        "threshold": final_threshold,
        "validation_metrics": results,
        "test_metrics": test_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": test_pred,
    }


def main(data_path: Path = RAW_DATA_PATH) -> None:
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    df = pd.read_csv(data_path)
    outcome = train_and_select(df)

    joblib.dump(
        {"pipeline": outcome["model"], "threshold": outcome["threshold"]},
        MODELS_DIR / "churn_model.pkl",
    )

    metrics_payload = {
        "selected_model": outcome["model_name"],
        "selected_threshold": outcome["threshold"],
        "validation_metrics": outcome["validation_metrics"],
        "test_metrics": outcome["test_metrics"],
    }

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    save_model_comparison(outcome["validation_metrics"], REPORTS_DIR / "model_comparison.json")
    save_report_markdown(outcome["validation_metrics"], REPORTS_DIR / "model_comparison.md")
    save_confusion_matrix(outcome["y_test"], outcome["test_pred"], REPORTS_DIR / "confusion_matrix.png")
    save_roc_pr_curves(
        outcome["y_test"],
        outcome["model"].predict_proba(outcome["X_test"])[:, 1],
        REPORTS_DIR / "roc_pr_curves.png",
    )

    print(f"Selected model: {outcome['model_name']}")
    print(json.dumps(outcome["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
