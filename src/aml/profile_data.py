"""Dataset inspection: shape, dtypes, summary statistics, missing values, imbalance.

Run:  $env:PYTHONPATH="src"; python -m aml.profile_data
Writes reports/dataset_profile.json and reports/dataset_preview.csv.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from aml.config import PROJECT_ROOT, load_config
from aml.data import load_raw


def profile(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Return a JSON-serialisable profile of the raw frame."""
    amount = df["Amount"].astype("float64")
    cats = [c for c in df.columns if str(df[c].dtype) in ("category", "object", "str", "string")]
    cats = [c for c in cats if c not in ("Time", "Date")]
    pos = int(df[target].sum())
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 1),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "amount_summary": {k: float(v) for k, v in amount.describe().items()},
        "categorical_cardinality": {c: int(df[c].nunique()) for c in cats},
        "target": {
            "name": target,
            "positives": pos,
            "negatives": int(len(df) - pos),
            "positive_rate_percent": round(100 * pos / len(df), 4),
            "imbalance_ratio_neg_to_pos": round((len(df) - pos) / max(pos, 1), 1),
        },
        "unique_senders": int(df["Sender_account"].nunique()),
        "unique_receivers": int(df["Receiver_account"].nunique()),
        "timestamp_range": [str(df["Timestamp"].min()), str(df["Timestamp"].max())],
    }


def main() -> None:
    target = load_config()["data"]["target"]
    df = load_raw()
    print(df.head(10).to_string())
    print(df.dtypes.to_string())
    report = profile(df, target)
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "dataset_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.head(10).to_csv(out / "dataset_preview.csv", index=False)
    print(
        json.dumps(
            {k: report[k] for k in ("shape", "memory_mb", "duplicate_rows", "target")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
