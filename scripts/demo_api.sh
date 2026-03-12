#!/usr/bin/env bash
set -euo pipefail

curl -s http://127.0.0.1:8000/health
echo

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id":"CUST-1001",
    "usage_frequency":6,
    "subscription_type":"monthly",
    "login_activity":5,
    "support_tickets":6,
    "payment_history":"failed",
    "avg_session_minutes":9.5,
    "monthly_spend":60,
    "tenure_months":3,
    "region":"NA"
  }'
echo

curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [
      {
        "customer_id":"CUST-1002",
        "usage_frequency":7,
        "subscription_type":"monthly",
        "login_activity":8,
        "support_tickets":5,
        "payment_history":"delayed",
        "avg_session_minutes":11,
        "monthly_spend":75,
        "tenure_months":5,
        "region":"EU"
      },
      {
        "customer_id":"CUST-1003",
        "usage_frequency":28,
        "subscription_type":"annual",
        "login_activity":35,
        "support_tickets":0,
        "payment_history":"on_time",
        "avg_session_minutes":24,
        "monthly_spend":95,
        "tenure_months":22,
        "region":"NA"
      }
    ]
  }'
echo
