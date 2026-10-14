"""Data cleaning with an auditable decision log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aml.logging_utils import get_logger

log = get_logger(__name__)


def clean(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply justified cleaning steps and return (clean_df, decision_log).

    Rows are only removed for exact duplicates, missing values in required fields,
    or non-positive amounts. Extreme amounts are retained: in AML data they can be
    genuine signal, and the largest values are handled by a log transform later.
    """
    report: dict[str, Any] = {"rows_in": int(len(df))}

    dup = int(df.duplicated().sum())
    report["duplicates_removed"] = dup
    if dup:
        df = df.drop_duplicates()

    missing = df.isna().sum()
    report["missing_by_column"] = {c: int(v) for c, v in missing.items() if v}
    required = [c for c in df.columns if c != "Laundering_type"]
    n_missing_rows = int(df[required].isna().any(axis=1).sum())
    report["rows_removed_missing"] = n_missing_rows
    if n_missing_rows:
        df = df.dropna(subset=required)

    nonpos = int((df["Amount"] <= 0).sum())
    report["rows_removed_nonpositive_amount"] = nonpos
    if nonpos:
        df = df[df["Amount"] > 0]

    amt = df["Amount"].astype("float64")
    q1, q3 = amt.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    flagged = amt > upper
    report["amount_outliers"] = {
        "iqr_upper_fence": float(upper),
        "count_above_fence": int(flagged.sum()),
        "positive_rate_above_fence": float(df.loc[flagged, target].mean()),
        "positive_rate_below_fence": float(df.loc[~flagged, target].mean()),
        "decision": "retained; log1p transform applied during feature engineering",
    }
    report["self_transfers"] = int((df["Sender_account"] == df["Receiver_account"]).sum())
    report["rows_out"] = int(len(df))
    log.info("Cleaning complete: %s -> %s rows", report["rows_in"], report["rows_out"])
    return df.reset_index(drop=True), report


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
