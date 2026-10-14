"""Chronological train/validation/test split and entity-overlap audit."""

from __future__ import annotations

from typing import Any

import pandas as pd


def time_split(
    df: pd.DataFrame, train_frac: float, valid_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted frame into contiguous train/valid/test blocks."""
    if not (0 < train_frac and 0 < valid_frac and train_frac + valid_frac < 1):
        raise ValueError("train_frac and valid_frac must be positive and sum to less than 1")
    if not df["Timestamp"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted by Timestamp before splitting")
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + valid_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def split_summary(parts: dict[str, pd.DataFrame], target: str) -> dict[str, Any]:
    train = parts["train"]
    seen_s, seen_r = set(train["Sender_account"].unique()), set(train["Receiver_account"].unique())
    out: dict[str, Any] = {}
    for name, p in parts.items():
        entry: dict[str, Any] = {
            "rows": int(len(p)),
            "positives": int(p[target].sum()),
            "positive_rate": float(p[target].mean()),
            "start": str(p["Timestamp"].min()),
            "end": str(p["Timestamp"].max()),
        }
        if name != "train":
            s_seen = p["Sender_account"].isin(seen_s)
            r_seen = p["Receiver_account"].isin(seen_r)
            entry["sender_seen_in_train"] = float(s_seen.mean())
            entry["receiver_seen_in_train"] = float(r_seen.mean())
            entry["both_unseen"] = float((~s_seen & ~r_seen).mean())
        out[name] = entry
    return out
