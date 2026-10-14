"""Model-specific preprocessing of engineered features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from aml.features import CATEGORICAL

BINARY = [
    "is_cross_border", "is_cross_currency", "is_round_amount", "in_structuring_band",
    "is_self_transfer", "sender_seen", "receiver_seen",
]  # fmt: skip


def build_preprocessor(kind: str, feature_columns: list[str]) -> ColumnTransformer:
    """`linear`: quantile-normal numerics plus one-hot. `tree`: raw numerics plus ordinal codes."""
    cats = [c for c in feature_columns if c in CATEGORICAL]
    nums = [c for c in feature_columns if c not in CATEGORICAL]
    if kind == "linear":
        cont = [c for c in nums if c not in BINARY]
        flags = [c for c in nums if c in BINARY]
        quant = QuantileTransformer(
            n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=42
        )
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts = [("cont", quant, cont), ("flags", "passthrough", flags), ("cat", encoder, cats)]
    elif kind == "tree":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts = [("num", "passthrough", nums), ("cat", encoder, cats)]
    else:
        raise ValueError(f"Unknown preprocessing kind: {kind}")
    return ColumnTransformer(parts, verbose_feature_names_out=False)
