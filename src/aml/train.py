"""Phase 4 runner. Stages: features, imbalance, tune, final.

Usage: python -m aml.train <stage> [--neg-pool N] [--cv-neg N]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from aml import evaluation as ev
from aml.cleaning import clean
from aml.config import PROJECT_ROOT, load_config
from aml.data import load_dataset
from aml.features import (
    CATEGORICAL,
    REQUIRED_INPUT,
    FeatureBuilder,
    build_causal_features,
    learn_categories,
)
from aml.logging_utils import get_logger
from aml.preprocessing import build_preprocessor

log = get_logger("aml.train")
SEED = 42
WINDOW_DAYS, BLOCK_DAYS = 28, 7
PROC = PROJECT_ROOT / "data" / "processed"
REP = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"
MODEL_NAMES = ["logreg", "tree", "forest"]
STRATEGIES = ["none", "class_weight", "random_over", "random_under", "smotenc"]
KIND = {"logreg": "linear", "tree": "tree", "forest": "tree"}
MODEL_LABELS = {"logreg": "Logistic Regression", "tree": "Decision Tree", "forest": "Random Forest"}


# ----------------------------------------------------------------------------- helpers
def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load_split(name: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols = _read_json(PROC / "feature_columns.json")["columns"]
    target = load_config()["data"]["target"]
    df = pd.read_parquet(PROC / f"{name}.parquet")
    return df[cols], df[target].to_numpy(), df[[target, "Laundering_type", "Timestamp"]]


def make_model(name: str, class_weight: str | None = None, **params: Any):
    if name == "logreg":
        base = {"C": 1.0, "max_iter": 1000}
        return LogisticRegression(**{**base, **params}, class_weight=class_weight)
    if name == "tree":
        base = {"max_depth": 8, "min_samples_leaf": 50}
        return DecisionTreeClassifier(
            **{**base, **params}, class_weight=class_weight, random_state=SEED
        )
    base = {"n_estimators": 80, "max_depth": 14, "min_samples_leaf": 10}
    cw = "balanced_subsample" if class_weight else None
    return RandomForestClassifier(
        **{**base, **params}, class_weight=cw, n_jobs=-1, random_state=SEED
    )


def make_pipeline(name: str, strategy: str, columns: list[str], **params: Any) -> ImbPipeline:
    if strategy == "smotenc" and name == "logreg":
        raise ValueError("SMOTENC is not applied to logistic regression (one-hot encoding)")
    prep = build_preprocessor(KIND[name], columns)
    steps: list[tuple[str, Any]] = [("prep", prep)]
    n_num = len([c for c in columns if c not in CATEGORICAL])
    cat_idx = list(range(n_num, n_num + len(CATEGORICAL)))
    if strategy == "random_over":
        steps.append(("sampler", RandomOverSampler(sampling_strategy=0.1, random_state=SEED)))
    elif strategy == "random_under":
        steps.append(("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)))
    elif strategy == "smotenc":
        steps.append(("under", RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)))
        steps.append(
            (
                "sampler",
                SMOTENC(categorical_features=cat_idx, sampling_strategy=0.5, random_state=SEED),
            )
        )
    cw = "balanced" if strategy == "class_weight" else None
    steps.append(("clf", make_model(name, cw, **params)))
    return ImbPipeline(steps)


def _scores(pipe, X) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- stage: features
def stage_features(neg_pool: int) -> None:
    cfg = load_config()
    target, sp = cfg["data"]["target"], cfg["split"]
    df, _ = clean(load_dataset(), target)
    n = len(df)
    i1 = int(n * sp["train_frac"])
    i2 = int(n * (sp["train_frac"] + sp["valid_frac"]))
    ts = df["Timestamp"]
    y = df[target].to_numpy()
    warm_end = ts.iloc[0] + pd.Timedelta(days=WINDOW_DAYS)
    ok = (ts >= warm_end).to_numpy()
    pos = np.where((y == 1) & ok & (np.arange(n) < i1))[0]
    neg_all = np.where((y == 0) & ok & (np.arange(n) < i1))[0]
    rng = np.random.default_rng(SEED)
    neg = rng.choice(neg_all, size=neg_pool, replace=False)
    pool = np.sort(np.concatenate([pos, neg]))
    valid, test = np.arange(i1, i2), np.arange(i2, n)

    train_df = df.iloc[:i1]
    cats = learn_categories(train_df)
    t0 = time.time()
    feats = build_causal_features(
        df[REQUIRED_INPUT], np.concatenate([pool, valid, test]), WINDOW_DAYS, BLOCK_DAYS, cats
    )
    log.info("Causal features built for %s rows in %.0fs", len(feats), time.time() - t0)
    types = df["Laundering_type"].astype(str)
    cols = [c for c in feats.columns]
    for name, idx in (("pool", pool), ("valid", valid), ("test", test)):
        part = feats.loc[feats.index.intersection(idx)].copy()
        part[target] = y[part.index]
        part["Laundering_type"] = types.iloc[part.index].to_numpy()
        part["Timestamp"] = ts.iloc[part.index].to_numpy()
        part.to_parquet(PROC / f"{name}.parquet", index=False)
        log.info("%s: %s rows, %s positives", name, len(part), int(part[target].sum()))

    # Static profile for the deployable artifact: trailing window at the end of validation.
    v_end = ts.iloc[i2 - 1]
    hist = df[(ts > v_end - pd.Timedelta(days=WINDOW_DAYS)) & (ts <= v_end)]
    joblib.dump(FeatureBuilder(cats).fit(hist[REQUIRED_INPUT]), PROC / "fb_static.pkl")
    _write_json(
        PROC / "feature_columns.json",
        {
            "columns": cols,
            "window_days": WINDOW_DAYS,
            "block_days": BLOCK_DAYS,
            "pool_negatives": int(neg_pool),
            "pool_positives": int(len(pos)),
            "warmup_end": str(warm_end),
            "static_profile_end": str(v_end),
        },
    )


# ----------------------------------------------------------------------------- stage: imbalance
def stage_imbalance() -> None:
    Xp, yp, _ = _load_split("pool")
    Xv, yv, _ = _load_split("valid")
    cols = list(Xp.columns)
    rows = []
    for name in MODEL_NAMES:
        for strat in STRATEGIES:
            if strat == "smotenc" and name == "logreg":
                continue
            t0 = time.time()
            pipe = make_pipeline(name, strat, cols).fit(Xp, yp)
            fit_s = time.time() - t0
            p = _scores(pipe, Xv)
            thr = ev.best_f1_threshold(yv, p)
            m = ev.threshold_metrics(yv, p, thr)
            row = {
                "model": name,
                "strategy": strat,
                **ev.ranking_metrics(yv, p),
                "best_f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "threshold": thr,
                "fit_seconds": round(fit_s, 1),
            }
            rows.append(row)
            log.info(
                "%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            )
            _write_json(REP / "imbalance_results.json", rows)
    res = pd.DataFrame(rows)
    res.to_csv(REP / "imbalance_results.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (REP / "figures").mkdir(parents=True, exist_ok=True)
    res.pivot(index="strategy", columns="model", values="pr_auc").reindex(STRATEGIES).plot.bar(
        figsize=(8, 4.5), title="Validation PR-AUC by imbalance strategy"
    )
    plt.ylabel("PR-AUC")
    plt.tight_layout()
    plt.savefig(REP / "figures" / "09_imbalance_pr_auc.png", dpi=130)
    plt.close()
    best = res.sort_values("pr_auc", ascending=False).groupby("model").head(1)
    _write_json(REP / "best_strategy.json", dict(zip(best["model"], best["strategy"], strict=True)))


# ----------------------------------------------------------------------------- stage: tune
SPACES: dict[str, tuple[dict[str, Any], int]] = {
    "logreg": ({"clf__C": loguniform(1e-3, 1e2)}, 6),
    "tree": (
        {
            "clf__max_depth": [4, 6, 8, 12, 16],
            "clf__min_samples_leaf": [10, 50, 200, 500],
            "clf__criterion": ["gini", "entropy"],
        },
        8,
    ),
    "forest": (
        {
            "clf__n_estimators": [60, 100],
            "clf__max_depth": [10, 14, 20],
            "clf__min_samples_leaf": [5, 20, 50],
            "clf__max_features": ["sqrt", 0.3],
        },
        5,
    ),
}


def stage_tune(cv_neg: int) -> None:
    Xp, yp, _ = _load_split("pool")
    strategies = _read_json(REP / "best_strategy.json")
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(
        np.where(yp == 0)[0], size=min(cv_neg, int((yp == 0).sum())), replace=False
    )
    idx = np.sort(np.concatenate([np.where(yp == 1)[0], neg_idx]))
    Xc, yc = Xp.iloc[idx], yp[idx]
    out = {}
    for name in MODEL_NAMES:
        space, n_iter = SPACES[name]
        pipe = make_pipeline(name, strategies[name], list(Xp.columns))
        search = RandomizedSearchCV(
            pipe,
            space,
            n_iter=n_iter,
            scoring="average_precision",
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            refit=False,
            random_state=SEED,
            n_jobs=1,
            error_score="raise",
        )
        t0 = time.time()
        search.fit(Xc, yc)
        params = {
            k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        }
        out[name] = {
            "strategy": strategies[name],
            "params": params,
            "cv_average_precision": float(search.best_score_),
            "cv_rows": int(len(yc)),
            "seconds": round(time.time() - t0),
        }
        log.info("tuned %s: %s", name, out[name])
        _write_json(REP / "tuning_results.json", out)


# ----------------------------------------------------------------------------- stage: final
def _importance(pipe: ImbPipeline, name: str) -> dict[str, float]:
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    vals = np.abs(clf.coef_[0]) if name == "logreg" else clf.feature_importances_
    top = pd.Series(vals, index=names).sort_values(ascending=False).head(15)
    return {k: float(v) for k, v in top.items()}


def stage_final() -> None:
    import matplotlib.pyplot as plt

    target = load_config()["data"]["target"]
    tuned = _read_json(REP / "tuning_results.json")
    Xp, yp, _ = _load_split("pool")
    Xv, yv, mv = _load_split("valid")
    Xt, yt, mt = _load_split("test")
    cols = list(Xp.columns)
    fitted: dict[str, ImbPipeline] = {}
    res: dict[str, Any] = {}
    sv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in MODEL_NAMES:
        t = tuned[name]
        pipe = make_pipeline(name, t["strategy"], cols, **t["params"]).fit(Xp, yp)
        fitted[name] = pipe
        pv = _scores(pipe, Xv)
        thr = ev.best_f1_threshold(yv, pv)
        thr80 = ev.threshold_for_recall(yv, pv, 0.80)
        res[name] = {
            "strategy": t["strategy"],
            "params": t["params"],
            "valid": {
                **ev.ranking_metrics(yv, pv),
                "at_best_f1": ev.threshold_metrics(yv, pv, thr),
                "at_recall_0.80": ev.threshold_metrics(yv, pv, thr80),
            },
        }
        sv[name] = (yv, pv)
        log.info(
            "valid %s: %s",
            name,
            {k: round(v, 4) for k, v in res[name]["valid"].items() if isinstance(v, float)},
        )
    best = max(MODEL_NAMES, key=lambda n: res[n]["valid"]["pr_auc"])
    log.info("Selected on validation PR-AUC: %s", best)

    # Single test evaluation, thresholds fixed from validation.
    st: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    conf: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        pt = _scores(fitted[name], Xt)
        thr = res[name]["valid"]["at_best_f1"]["threshold"]
        thr80 = res[name]["valid"]["at_recall_0.80"]["threshold"]
        res[name]["test"] = {
            **ev.ranking_metrics(yt, pt),
            "at_best_f1": ev.threshold_metrics(yt, pt, thr),
            "at_recall_0.80": ev.threshold_metrics(yt, pt, thr80),
        }
        st[name] = (yt, pt)
        conf[name] = res[name]["test"]["at_best_f1"]
    res["selected"] = best
    grid = np.linspace(0.02, 0.98, 49)
    sweep = {
        name: {
            "valid": ev.threshold_sweep(*sv[name], grid),
            "test": ev.threshold_sweep(*st[name], grid),
        }
        for name in MODEL_NAMES
    }
    _write_json(REP / "threshold_sweep.json", sweep)
    ev.plot_sweeps(
        {MODEL_LABELS[n]: sweep[n]["valid"] for n in MODEL_NAMES},
        sweep[best]["test"],
        MODEL_LABELS[best],
        res[best]["valid"]["at_best_f1"]["threshold"],
        REP / "figures" / "14_threshold_sweep.png",
    )
    ev.plot_curves(sv, REP / "figures" / "10_curves_valid.png", "validation")
    ev.plot_curves(st, REP / "figures" / "11_curves_test.png", "test")
    ev.plot_confusions(
        conf, REP / "figures" / "12_confusion_test.png", "Test set, validation-chosen threshold"
    )
    imp = _importance(fitted[best], best)
    res["feature_importance_selected"] = imp
    pd.Series(imp).sort_values().plot.barh(figsize=(7, 5), title=f"Top features - {best}")
    plt.tight_layout()
    plt.savefig(REP / "figures" / "13_feature_importance.png", dpi=130)
    plt.close()

    # Error analysis for the selected model on test (typology used for analysis only).
    pt = st[best][1]
    thr = res[best]["valid"]["at_best_f1"]["threshold"]
    d = mt.assign(
        pred=(pt >= thr).astype(int), Payment_type=Xt["Payment_type"].astype(str).to_numpy()
    )
    pos = d[d[target] == 1].groupby("Laundering_type")["pred"].agg(["size", "mean"])
    res["error_analysis"] = {
        "recall_by_typology_test": {
            k: {"n": int(r["size"]), "recall": round(float(r["mean"]), 3)}
            for k, r in pos.sort_values("mean").iterrows()
        },
        "false_positives_by_payment_type": d[(d[target] == 0) & (d.pred == 1)]["Payment_type"]
        .value_counts()
        .to_dict(),
    }

    # Deployable artifact: static profile + fitted preprocessing + fitted classifier.
    fb_static = joblib.load(PROC / "fb_static.pkl")
    final = Pipeline(
        [
            ("features", fb_static),
            ("prep", fitted[best].named_steps["prep"]),
            ("clf", fitted[best].named_steps["clf"]),
        ]
    )
    MODELS.mkdir(exist_ok=True)
    path = MODELS / "aml_pipeline.pkl"
    joblib.dump(final, path)
    meta = {
        "model": best,
        "strategy": tuned[best]["strategy"],
        "params": tuned[best]["params"],
        "threshold": res[best]["valid"]["at_best_f1"]["threshold"],
        "threshold_rule": "maximise F1 on the time-ordered validation split",
        "input_columns": REQUIRED_INPUT,
        "window_days": WINDOW_DAYS,
        "block_days": BLOCK_DAYS,
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": "3.11.9 (sandbox)",
        },
    }
    _write_json(MODELS / "model_meta.json", meta)
    _write_json(REP / "model_results.json", res)
    log.info("Saved %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "imbalance", "tune", "final"])
    ap.add_argument("--neg-pool", type=int, default=200_000)
    ap.add_argument("--cv-neg", type=int, default=60_000)
    a = ap.parse_args()
    if a.stage == "features":
        stage_features(a.neg_pool)
    elif a.stage == "imbalance":
        stage_imbalance()
    elif a.stage == "tune":
        stage_tune(a.cv_neg)
    else:
        stage_final()


if __name__ == "__main__":
    main()
