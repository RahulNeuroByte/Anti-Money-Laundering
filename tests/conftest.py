import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from aml.features import FeatureBuilder, learn_categories
from aml.preprocessing import build_preprocessor


def synthetic_transactions(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.integers(0, 30 * 86400, n))
    return pd.DataFrame(
        {
            "Timestamp": pd.Timestamp("2023-01-01") + pd.to_timedelta(secs, "s"),
            "Sender_account": rng.integers(0, 50, n),
            "Receiver_account": rng.integers(0, 70, n),
            "Amount": rng.uniform(5, 20000, n),
            "Payment_type": rng.choice(["ACH", "Cheque", "Cross-border"], n),
            "Payment_currency": "UK pounds",
            "Received_currency": rng.choice(["UK pounds", "Euro"], n),
            "Sender_bank_location": "UK",
            "Receiver_bank_location": rng.choice(["UK", "Spain"], n),
        }
    )


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A small fitted pipeline with the same structure as the real artifact."""
    df = synthetic_transactions()
    y = (df["Amount"] > 15000).astype(int).to_numpy()
    fb = FeatureBuilder(learn_categories(df)).fit(df)
    feats = fb.transform(df)
    prep = build_preprocessor("tree", list(feats.columns)).fit(feats)
    clf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0)
    clf.fit(prep.transform(feats), y)
    pipe = Pipeline([("features", fb), ("prep", prep), ("clf", clf)])
    folder = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, folder / "aml_pipeline.pkl")
    (folder / "model_meta.json").write_text(json.dumps({"model": "tiny", "threshold": 0.5}))
    return folder / "aml_pipeline.pkl"
