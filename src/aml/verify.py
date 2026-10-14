"""Artifact validation and converged logistic-regression imbalance rerun."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

from aml import evaluation as ev
from aml.cleaning import clean
from aml.config import PROJECT_ROOT, load_config
from aml.data import load_dataset
from aml.features import REQUIRED_INPUT
from aml.train import STRATEGIES, _load_split, _scores, make_pipeline


def validate_artifact() -> dict:
    cfg = load_config()
    target, sp = cfg["data"]["target"], cfg["split"]
    meta = json.loads((PROJECT_ROOT / "models" / "model_meta.json").read_text())
    pipe = joblib.load(PROJECT_ROOT / "models" / "aml_pipeline.pkl")
    df, _ = clean(load_dataset(), target)
    n = len(df)
    i2 = int(n * (sp["train_frac"] + sp["valid_frac"]))
    test = df.iloc[i2:]
    raw, y = test[REQUIRED_INPUT], test[target].to_numpy()

    sample = raw.iloc[:20_000]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "copy.pkl"
        joblib.dump(pipe, path)
        reloaded = joblib.load(path)
    p_mem = pipe.predict_proba(sample)[:, 1]
    p_rel = reloaded.predict_proba(sample)[:, 1]
    parity = float(np.max(np.abs(p_mem - p_rel)))

    t0 = time.time()
    p = pipe.predict_proba(raw)[:, 1]
    secs = time.time() - t0
    thr = meta["threshold"]
    out = {
        "reload_max_abs_diff_20k_rows": parity,
        "reload_exact_match": bool(parity == 0.0),
        "raw_test_scoring_seconds": round(secs, 1),
        "static_profile_end": json.loads(
            (PROJECT_ROOT / "data" / "processed" / "feature_columns.json").read_text()
        )["static_profile_end"],
        "test_with_static_profile": {
            **ev.ranking_metrics(y, p),
            "at_saved_threshold": ev.threshold_metrics(y, p, thr),
        },
    }
    (PROJECT_ROOT / "reports" / "artifact_validation.json").write_text(json.dumps(out, indent=2))
    return out


def rerun_logreg() -> list[dict]:
    Xp, yp, _ = _load_split("pool")
    Xv, yv, _ = _load_split("valid")
    rows = []
    for strat in [s for s in STRATEGIES if s != "smotenc"]:
        pipe = make_pipeline("logreg", strat, list(Xp.columns)).fit(Xp, yp)
        p = _scores(pipe, Xv)
        rows.append({"strategy": strat, **ev.ranking_metrics(yv, p)})
    (PROJECT_ROOT / "reports" / "logreg_converged.json").write_text(json.dumps(rows, indent=2))
    return rows


def auc_mismatch() -> dict:
    """Univariate AUC of whole-window profile features: training sample versus validation sample.

    Documents why causal (lagged) profiles are used: profiles fitted on the whole training window
    look stronger on training rows than on later validation rows.
    """
    from sklearn.metrics import roc_auc_score

    from aml.features import CATEGORICAL, FeatureBuilder
    from aml.splitting import time_split

    cfg = load_config()
    target, sp = cfg["data"]["target"], cfg["split"]
    df, _ = clean(load_dataset(), target)
    train, valid, _ = time_split(df, sp["train_frac"], sp["valid_frac"])
    train_auc = json.loads((PROJECT_ROOT / "reports" / "eda_summary.json").read_text())[
        "univariate_auc_train_sample"
    ]
    fb = FeatureBuilder().fit(train)
    rng = np.random.default_rng(42)
    pos = valid.index[valid[target] == 1]
    neg = rng.choice(valid.index[valid[target] == 0], 500_000, replace=False)
    sample = valid.loc[np.sort(np.concatenate([pos, neg]))]
    feats = fb.transform(sample).drop(columns=CATEGORICAL)
    y = sample[target].to_numpy()
    rows = [
        (c, train_auc[c], float(roc_auc_score(y, feats[c])))
        for c in train_auc
        if feats[c].nunique() > 1
    ]
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    out = {c: [round(a, 4), round(b, 4)] for c, a, b in rows}
    (PROJECT_ROOT / "reports" / "train_vs_valid_auc.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    actions = {"artifact": validate_artifact, "logreg": rerun_logreg, "auc": auc_mismatch}
    print(json.dumps(actions[which](), indent=1))
