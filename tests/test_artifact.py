import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from aml.features import REQUIRED_INPUT

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "models" / "aml_pipeline.pkl"
pytestmark = pytest.mark.skipif(not PKL.exists(), reason="model artifact not built")


def _row(**over):
    base = {
        "Timestamp": pd.Timestamp("2023-08-01 12:00:00"), "Sender_account": 1,
        "Receiver_account": 2, "Amount": 1500.0, "Payment_type": "ACH",
        "Payment_currency": "UK pounds", "Received_currency": "UK pounds",
        "Sender_bank_location": "UK", "Receiver_bank_location": "UK",
    }  # fmt: skip
    return pd.DataFrame([{**base, **over}])[REQUIRED_INPUT]


def test_pipeline_scores_in_unit_interval_and_handles_unseen_values():
    pipe = joblib.load(PKL)
    rows = pd.concat([_row(), _row(Payment_type="Crypto", Sender_account=-5)], ignore_index=True)
    p = pipe.predict_proba(rows)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_meta_threshold_is_probability():
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    assert 0 < meta["threshold"] < 1
