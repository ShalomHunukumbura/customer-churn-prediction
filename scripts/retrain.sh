#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m src.data.generate_dataset
python -m src.eda
python -m src.train
