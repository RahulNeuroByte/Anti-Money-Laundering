import numpy as np
import pandas as pd

from aml.features import build_causal_features, learn_categories


def _timeline(n=4000, days=70, seed=0):
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.integers(0, days * 86400, n))
    ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(secs, "s")
    return pd.DataFrame(
        {
            "Timestamp": ts,
            "Sender_account": rng.integers(0, 60, n),
            "Receiver_account": rng.integers(0, 80, n),
            "Amount": rng.uniform(5, 5000, n),
            "Payment_type": rng.choice(["ACH", "Cheque"], n),
            "Payment_currency": "UK pounds",
            "Received_currency": rng.choice(["UK pounds", "Euro"], n),
            "Sender_bank_location": "UK",
            "Receiver_bank_location": rng.choice(["UK", "Spain"], n),
        }
    )


def test_features_do_not_depend_on_future_rows():
    df = _timeline()
    cats = learn_categories(df)
    rows = np.arange(len(df))
    base = build_causal_features(df, rows, 14, 7, cats)
    cut = pd.Timestamp("2023-02-20")
    changed = df.copy()
    future = changed["Timestamp"] >= cut
    changed.loc[future, "Amount"] *= 7.0
    changed.loc[future, "Sender_account"] = 999
    alt = build_causal_features(changed, rows, 14, 7, cats)
    past = df.index[df["Timestamp"] < cut]
    past = past.intersection(base.index)
    cols = ["s_out_count", "s_out_total", "r_in_nuniq", "receiver_flow_ratio"]
    pd.testing.assert_frame_equal(base.loc[past, cols], alt.loc[past, cols])


def test_warmup_rows_excluded_and_no_nan():
    df = _timeline()
    out = build_causal_features(df, np.arange(len(df)), 14, 7, learn_categories(df))
    first_ok = df["Timestamp"].iloc[0] + pd.Timedelta(days=14)
    assert (df.loc[out.index, "Timestamp"] >= first_ok).all()
    assert len(out) < len(df)
    assert out["s_out_count"].max() > 0
