import pandas as pd
import pytest

from aml.cleaning import clean
from aml.features import CATEGORICAL, FeatureBuilder
from aml.splitting import time_split


def _frame():
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2023-01-01", periods=6, freq="h"),
            "Sender_account": [1, 1, 2, 3, 4, 9],
            "Receiver_account": [2, 3, 3, 1, 1, 8],
            "Amount": [100.0, 9500.0, 50.0, 20.0, 700.0, 10.0],
            "Payment_type": ["ACH", "Cross-border", "ACH", "ACH", "Cheque", "ACH"],
            "Payment_currency": ["UK pounds"] * 6,
            "Received_currency": ["UK pounds", "Euro", *["UK pounds"] * 4],
            "Sender_bank_location": ["UK"] * 6,
            "Receiver_bank_location": ["UK", "Spain", "UK", "UK", "UK", "UK"],
            "Is_laundering": [0, 1, 0, 0, 0, 0],
            "Laundering_type": ["Normal_Fan_In"] * 6,
        }
    )


def test_unseen_accounts_flagged_and_no_nan():
    df = _frame()
    fb = FeatureBuilder().fit(df.iloc[:4])
    out = fb.transform(df.iloc[4:])
    assert out.loc[1, "sender_seen"] == 0 and out.loc[1, "receiver_seen"] == 0
    assert not out.drop(columns=CATEGORICAL).isna().any().any()
    assert "Laundering_type" not in out.columns and "Is_laundering" not in out.columns


def test_cross_flags_and_band():
    out = FeatureBuilder().fit(_frame()).transform(_frame())
    assert out.loc[1, "is_cross_border"] == 1 and out.loc[1, "is_cross_currency"] == 1
    assert out.loc[1, "in_structuring_band"] == 1 and out.loc[0, "in_structuring_band"] == 0


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureBuilder().transform(_frame())


def test_missing_column_raises():
    with pytest.raises(ValueError):
        FeatureBuilder().fit(_frame().drop(columns=["Amount"]))


def test_time_split_is_chronological_and_rejects_unsorted():
    df = _frame()
    tr, va, te = time_split(df, 0.5, 0.25)
    assert tr["Timestamp"].max() <= va["Timestamp"].min() <= te["Timestamp"].min()
    with pytest.raises(ValueError):
        time_split(df.iloc[::-1], 0.5, 0.25)


def test_clean_drops_duplicates_and_nonpositive():
    df = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    df.loc[2, "Amount"] = -5.0
    out, rep = clean(df, "Is_laundering")
    assert rep["duplicates_removed"] == 1 and rep["rows_removed_nonpositive_amount"] == 1
    assert len(out) == len(df) - 2


def test_unseen_category_mapped_not_error():
    import warnings

    df = _frame()
    fb = FeatureBuilder().fit(df)
    new = df.iloc[[0]].copy()
    new["Payment_type"] = "Crypto"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = fb.transform(new)
    assert out.loc[0, "Payment_type"] == "__unseen__"


def test_time_split_bad_fractions():
    with pytest.raises(ValueError):
        time_split(_frame(), 0.9, 0.2)
