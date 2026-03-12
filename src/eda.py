from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, REPORTS_DIR
from src.utils import ensure_dir


def build_eda_summary(data_path: Path = RAW_DATA_PATH, output_path: Path = REPORTS_DIR / "eda_summary.md") -> None:
    df = pd.read_csv(data_path)
    ensure_dir(output_path.parent)

    churn_rate = float(df["churn"].mean())
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)

    by_subscription = (
        df.groupby("subscription_type", dropna=False)["churn"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )

    by_payment = (
        df.groupby("payment_history", dropna=False)["churn"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )

    lines = [
        "# EDA Summary",
        "",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        f"- Churn rate: {churn_rate:.3f}",
        "",
        "## Missing Value % by Column",
        "",
    ]

    for col, pct in missing.items():
        lines.append(f"- {col}: {pct}%")

    lines.extend(["", "## Churn by Subscription Type", ""])
    for idx, row in by_subscription.iterrows():
        lines.append(f"- {idx}: count={int(row['count'])}, churn_rate={row['churn_rate']:.3f}")

    lines.extend(["", "## Churn by Payment History", ""])
    for idx, row in by_payment.iterrows():
        lines.append(f"- {idx}: count={int(row['count'])}, churn_rate={row['churn_rate']:.3f}")

    output_path.write_text("\n".join(lines))
    print(f"Wrote EDA summary to {output_path}")


if __name__ == "__main__":
    build_eda_summary()
