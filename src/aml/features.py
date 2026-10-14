"""Feature engineering: row-level features plus account profiles fitted on training data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL = [
    "Payment_type",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
]
REQUIRED_INPUT = [
    "Timestamp",
    "Sender_account",
    "Receiver_account",
    "Amount",
    *CATEGORICAL,
]
STRUCTURING_BAND = (9000.0, 10000.0)
UNSEEN = "__unseen__"
LOG_COLUMNS = ("s_out_total", "s_in_total", "r_in_total", "r_out_total", "s_out_max", "r_in_max")


def _profile(df: pd.DataFrame, own: str, other: str, prefix: str, with_stats: bool) -> pd.DataFrame:
    amt = df["Amount"].astype("float64")
    g = amt.groupby(df[own])
    out = pd.DataFrame({f"{prefix}_count": g.size(), f"{prefix}_total": g.sum()})
    if with_stats:
        out[f"{prefix}_mean"] = g.mean()
        out[f"{prefix}_max"] = g.max()
    out[f"{prefix}_nuniq"] = df.groupby(own)[other].nunique()
    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds model features. `fit` learns per-account profiles from the training window only.

    Accounts not seen during fit receive zero profile values and a `*_seen` flag of 0,
    which is also the behaviour at inference time for new accounts.
    """

    def __init__(self, categories: dict[str, list[str]] | None = None):
        self.categories = categories

    def fit(self, X: pd.DataFrame, y=None) -> FeatureBuilder:
        missing = [c for c in REQUIRED_INPUT if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        s, r = "Sender_account", "Receiver_account"
        self.sender_out_ = _profile(X, s, r, "s_out", True)
        self.sender_in_ = _profile(X.rename(columns={r: "k", s: "o"}), "k", "o", "s_in", False)
        self.receiver_in_ = _profile(X, r, s, "r_in", True)
        self.receiver_out_ = _profile(X.rename(columns={s: "k", r: "o"}), "k", "o", "r_out", False)
        base = self.categories or {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAL}
        self.categories_ = {c: [*base[c], UNSEEN] for c in CATEGORICAL}
        return self

    def _lookup(self, table: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
        return table.reindex(keys.to_numpy()).reset_index(drop=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "sender_out_"):
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        missing = [c for c in REQUIRED_INPUT if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        amt = X["Amount"].astype("float64").to_numpy()
        ts = pd.to_datetime(X["Timestamp"])
        out = pd.DataFrame(index=range(len(X)))
        out["log_amount"] = np.log1p(amt)
        out["hour"] = ts.dt.hour.to_numpy()
        out["day_of_week"] = ts.dt.dayofweek.to_numpy()
        out["is_cross_border"] = (
            X["Sender_bank_location"].astype(str).to_numpy()
            != X["Receiver_bank_location"].astype(str).to_numpy()
        ).astype("int8")
        out["is_cross_currency"] = (
            X["Payment_currency"].astype(str).to_numpy()
            != X["Received_currency"].astype(str).to_numpy()
        ).astype("int8")
        out["is_round_amount"] = (np.mod(amt, 100) == 0).astype("int8")
        lo, hi = STRUCTURING_BAND
        out["in_structuring_band"] = ((amt >= lo) & (amt < hi)).astype("int8")
        out["is_self_transfer"] = (
            X["Sender_account"].to_numpy() == X["Receiver_account"].to_numpy()
        ).astype("int8")

        so = self._lookup(self.sender_out_, X["Sender_account"])
        si = self._lookup(self.sender_in_, X["Sender_account"])
        ri = self._lookup(self.receiver_in_, X["Receiver_account"])
        ro = self._lookup(self.receiver_out_, X["Receiver_account"])
        out["sender_seen"] = so["s_out_count"].notna().astype("int8").to_numpy()
        out["receiver_seen"] = ri["r_in_count"].notna().astype("int8").to_numpy()
        profiles = pd.concat([so, si, ri, ro], axis=1)
        out = pd.concat([out, profiles.fillna(0.0).astype("float64")], axis=1)

        out["sender_flow_ratio"] = out["s_in_total"] / (out["s_out_total"] + 1.0)
        out["receiver_flow_ratio"] = out["r_out_total"] / (out["r_in_total"] + 1.0)
        ratios = (("amount_to_sender_mean", "s_out_mean"), ("amount_to_receiver_mean", "r_in_mean"))
        for new, col in ratios:
            denom = out[col].to_numpy(dtype="float64")
            out[new] = np.where(denom > 0, amt / np.where(denom > 0, denom, 1.0), 1.0)
        for c in LOG_COLUMNS:
            out[c] = np.log1p(out[c])
        for c in CATEGORICAL:
            vals = X[c].astype(str)
            vals = vals.where(vals.isin(self.categories_[c]), UNSEEN)
            out[c] = pd.Categorical(vals.to_numpy(), categories=self.categories_[c])
        num = out.select_dtypes("float64").columns
        out[num] = out[num].astype("float32")
        return out


def learn_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Category levels for the categorical features, fixed once from the training window."""
    return {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}


def build_causal_features(
    df: pd.DataFrame,
    rows: np.ndarray,
    window_days: int = 28,
    block_days: int = 7,
    categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Features for the positional `rows` of a time-sorted frame, using only past transactions.

    Time is cut into blocks of `block_days`. Account profiles for a block are built from the
    `window_days` immediately before the block start, so no row sees its own or future activity,
    and every row sees a history of the same length. Rows in the first `window_days` (warm-up)
    have no full history and are not returned. No labels are used.
    """
    ts = df["Timestamp"].to_numpy()
    if not np.all(ts[:-1] <= ts[1:]):
        raise ValueError("Frame must be sorted by Timestamp")
    rows = np.sort(np.asarray(rows))
    win, blk = np.timedelta64(window_days, "D"), np.timedelta64(block_days, "D")
    parts = []
    edge = ts[0] + win
    while edge <= ts[-1]:
        lo = np.searchsorted(ts, edge - win, "left")
        mid = np.searchsorted(ts, edge, "left")
        hi = np.searchsorted(ts, edge + blk, "left")
        a, z = np.searchsorted(rows, mid, "left"), np.searchsorted(rows, hi, "left")
        if z > a:
            fb = FeatureBuilder(categories).fit(df.iloc[lo:mid])
            feats = fb.transform(df.iloc[rows[a:z]])
            feats.index = rows[a:z]
            parts.append(feats)
        edge = edge + blk
    return pd.concat(parts)
