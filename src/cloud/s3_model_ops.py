from __future__ import annotations

import os
from pathlib import Path


def upload_model_to_s3(model_path: Path, metrics_path: Path) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Install it before using S3 upload.") from exc

    bucket = os.getenv("AWS_S3_BUCKET")
    prefix = os.getenv("AWS_S3_PREFIX", "churn-model")

    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET environment variable is required")

    s3 = boto3.client("s3")
    s3.upload_file(str(model_path), bucket, f"{prefix}/churn_model.pkl")
    s3.upload_file(str(metrics_path), bucket, f"{prefix}/metrics.json")
    print(f"Uploaded model artifacts to s3://{bucket}/{prefix}/")


if __name__ == "__main__":
    upload_model_to_s3(Path("models/churn_model.pkl"), Path("models/metrics.json"))
