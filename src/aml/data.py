"""Typed dataset loading with a parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml.config import PROJECT_ROOT, load_config
from aml.logging_utils import get_logger

log = get_logger(__name__)

RAW_DTYPES = {
    "Sender_account": "int64",
    "Receiver_account": "int64",
    "Amount": "float32",
    "Payment_currency": "category",
    "Received_currency": "category",
    "Sender_bank_location": "category",
    "Receiver_bank_location": "category",
    "Payment_type": "category",
    "Is_laundering": "int8",
    "Laundering_type": "category",
}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_raw(csv_path: str | None = None) -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes and add a single `Timestamp` column."""
    cfg = load_config()
    path = _resolve(csv_path or cfg["data"]["raw_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at {path}. See README for download steps.")
    df = pd.read_csv(path, dtype={**RAW_DTYPES, "Time": "string", "Date": "string"})
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S")
    return df.drop(columns=["Date", "Time"])


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    """Return the dataset sorted by time, using a parquet cache in data/processed."""
    cfg = load_config()
    cache = _resolve(cfg["data"]["processed_dir"]) / "saml_d.parquet"
    if cache.is_file() and not refresh:
        log.info("Loading cached parquet: %s", cache)
        return pd.read_parquet(cache)
    log.info("Reading raw CSV and writing cache")
    df = load_raw().sort_values("Timestamp", kind="stable").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df
