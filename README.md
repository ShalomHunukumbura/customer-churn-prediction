# Associate AI Engineer Technical Assessment

Option selected: **Option 2 - Customer Churn Prediction**

This repository contains a full, practical churn-prediction ML system with:
- Synthetic data generation (7,000+ rows, realistic imbalance, missing values)
- Preprocessing and feature engineering
- Multi-model training and comparison
- Threshold tuning for churn-class F1
- FastAPI inference service (`/predict` and `/predict/batch`)
- Evaluation artifacts and operations plan

## 1. Problem Statement
Predict churn risk for SaaS customers using:
- `usage_frequency`
- `subscription_type`
- `login_activity`
- `support_tickets`
- `payment_history`

Additional supporting features are included (`avg_session_minutes`, `monthly_spend`, `tenure_months`, `region`) to better simulate real business data.

## 2. Repository Layout
- `src/data/generate_dataset.py`: Creates synthetic dataset with controlled missingness
- `src/eda.py`: Generates EDA summary report
- `src/data_preprocessing.py`: Imputation, encoding, scaling, feature engineering
- `src/train.py`: Trains 3 models, tunes threshold, saves artifacts
- `src/evaluate.py`: Confusion matrix + ROC/PR curve generation
- `src/predict.py`: Single and batch prediction logic
- `src/api.py`: FastAPI application
- `src/cloud/s3_model_ops.py`: Optional S3 upload helper
- `scripts/retrain.sh`: End-to-end retraining script
- `scripts/demo_api.sh`: Curl-based API demonstration script
- `docs/s3_evidence.md`: S3 artifact upload evidence and verification output
- `tests/`: Lightweight validation tests

## 3. Data Design
Generated dataset properties:
- 7,000 records minimum
- Numerical + categorical features
- Missing values injected in multiple columns
- Realistic churn drivers:
  - payment issues
  - high support tickets
  - low engagement
  - lower tenure and monthly plans

Output target:
- `churn` (`0` or `1`)

## 4. Preprocessing & Feature Engineering
Preprocessing:
- Numerical imputation: median
- Categorical imputation: most frequent
- Categorical encoding: one-hot (`handle_unknown='ignore'`)
- Scaling: standard scaler for numerical features

Feature engineering:
- `ticket_per_login`
- `engagement_score`
- `payment_risk_score`
- `high_support_flag`
- `low_tenure_flag`

## 5. Modeling Approach
Models trained:
- Logistic Regression
- Random Forest
- HistGradientBoosting

Training flow:
- Stratified split: 70% train / 15% val / 15% test
- Class imbalance handled with balanced sample weights
- Threshold tuning on validation set for best churn-class F1
- Final model selected by validation F1 + PR-AUC tie-break

Saved artifacts:
- `models/churn_model.pkl` (pipeline + threshold)
- `models/metrics.json`
- `reports/confusion_matrix.png`
- `reports/roc_pr_curves.png`
- `reports/model_comparison.json`
- `reports/model_comparison.md`
- `reports/eda_summary.md`

## 6. API Endpoints
- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /model/info`

### Example: `POST /predict`
```json
{
  "customer_id": "CUST-1001",
  "usage_frequency": 6,
  "subscription_type": "monthly",
  "login_activity": 5,
  "support_tickets": 6,
  "payment_history": "failed",
  "avg_session_minutes": 9.5,
  "monthly_spend": 60,
  "tenure_months": 3,
  "region": "NA"
}
```

## 7. Local Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.generate_dataset
python -m src.eda
python -m src.train
pytest -q
uvicorn src.api:app --reload
```

Or with Makefile:
```bash
make setup
make all
make serve
```

## 8. AWS S3 bucket
- Set `AWS_S3_BUCKET` and optional `AWS_S3_PREFIX`
- Upload artifacts:
```bash
python -m src.cloud.s3_model_ops
```
- Verify uploaded artifacts:
```bash
aws s3 ls s3://<your-bucket>/<your-prefix>/
```
- Proof details in `docs/s3_evidence.md`

## 9. Deployment Notes
- Dockerfile is included for containerized serving
- Retraining script is included: `scripts/retrain.sh`
- Detailed monitoring/retraining/scaling/cost strategy: `docs/operations_plan.md`

## 10. Optimization & Scaling Explanation
### Scale to 100k+ records
- Use a hybrid design: real-time API for single requests and batch jobs for bulk scoring.
- Run multiple API workers behind a load balancer/autoscaling group.
- Store batch inputs in columnar format (Parquet) and process asynchronously via queue workers.

### Retraining approach
- Retrain on a schedule (monthly or weekly) using newly labeled outcomes.
- Use evaluation gates (minimum churn-class F1/PR-AUC) before promoting a new model.
- Version artifacts and keep rollback capability to the previous stable model.

### Monitoring approach
- API monitoring: p95 latency, error rate, throughput.
- Model/data monitoring: feature drift, prediction distribution drift, and post-label metric drift.
- Alert when thresholds are breached (for example, latency spike or significant F1 drop).

### Cloud cost considerations
- Keep training as scheduled jobs (avoid always-on compute).
- Store artifacts in S3 and load them at service startup.
- Use low-cost compute tiers for dev/test and spot/preemptible instances for batch workloads.
