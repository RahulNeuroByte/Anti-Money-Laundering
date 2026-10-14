"""Phase 6: log the three models to MLflow, compare runs, register and alias the best.

Run from the project root:  $env:PYTHONPATH="src"; python -m mlops.track_models
Tracking store: MLFLOW_TRACKING_URI (default sqlite:///mlflow.db). The file store is deprecated.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from aml import evaluation as ev
from aml.config import PROJECT_ROOT, load_config
from aml.features import CATEGORICAL, REQUIRED_INPUT
from aml.inference import load_artifact, score
from aml.train import (
    MODEL_NAMES,
    PROC,
    REP,
    _importance,
    _load_split,
    _read_json,
    _scores,
    make_pipeline,
)

EXPERIMENT = "aml-detection"
REGISTERED_NAME = "aml-detector"
PIP_REQUIREMENTS = [
    f"scikit-learn=={sklearn.__version__}",
    f"pandas=={pd.__version__}",
    f"numpy=={np.__version__}",
    "joblib",
    "pyyaml",
]


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()

    tuned = _read_json(REP / "tuning_results.json")
    phase4 = _read_json(REP / "model_results.json")
    fcols = _read_json(PROC / "feature_columns.json")
    Xp, yp, _ = _load_split("pool")
    Xv, yv, _ = _load_split("valid")
    Xt, yt, _ = _load_split("test")
    fb_static = joblib.load(PROC / "fb_static.pkl")
    raw_csv = PROJECT_ROOT / cfg["data"]["raw_path"]
    data_hash = sha256_of(raw_csv) if raw_csv.is_file() else "unavailable"
    full = pd.read_parquet(PROC / "saml_d.parquet", columns=REQUIRED_INPUT)
    # Callers send JSON numbers (float64 / int64); the signature must accept those, not float32.
    raw_example = (
        full.iloc[-50:-47]
        .reset_index(drop=True)
        .astype({"Amount": "float64", "Sender_account": "int64", "Receiver_account": "int64"})
        .astype({c: "object" for c in CATEGORICAL})
    )
    # JSON clients send ISO-8601 strings; FeatureBuilder parses them with pd.to_datetime.
    raw_example["Timestamp"] = raw_example["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    del full

    run_ids: dict[str, str] = {}
    for name in MODEL_NAMES:
        t = tuned[name]
        fitted = make_pipeline(name, t["strategy"], list(Xp.columns), **t["params"]).fit(Xp, yp)
        pv, pt = _scores(fitted, Xv), _scores(fitted, Xt)
        thr = ev.best_f1_threshold(yv, pv)
        mv, mt = ev.threshold_metrics(yv, pv, thr), ev.threshold_metrics(yt, pt, thr)
        rank_v, rank_t = ev.ranking_metrics(yv, pv), ev.ranking_metrics(yt, pt)
        ref = phase4[name]
        drift = max(
            abs(rank_v["pr_auc"] - ref["valid"]["pr_auc"]),
            abs(rank_t["pr_auc"] - ref["test"]["pr_auc"]),
            abs(mt["f1"] - ref["test"]["at_best_f1"]["f1"]),
        )
        deploy = Pipeline(
            [
                ("features", fb_static),
                ("prep", fitted.named_steps["prep"]),
                ("clf", fitted.named_steps["clf"]),
            ]
        )
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "model": name,
                    "imbalance_strategy": t["strategy"],
                    **{f"hp_{k}": v for k, v in t["params"].items()},
                    "pool_positives": fcols["pool_positives"],
                    "pool_negatives": fcols["pool_negatives"],
                    "window_days": fcols["window_days"],
                    "block_days": fcols["block_days"],
                    "split": f"{cfg['split']['train_frac']}/{cfg['split']['valid_frac']}/time",
                    "threshold_rule": "max F1 on validation",
                    "data_sha256": data_hash,
                    "random_state": 42,
                }
            )
            for split, rank, m in (("valid", rank_v, mv), ("test", rank_t, mt)):
                mlflow.log_metrics(
                    {
                        f"{split}_roc_auc": rank["roc_auc"],
                        f"{split}_pr_auc": rank["pr_auc"],
                        f"{split}_precision": m["precision"],
                        f"{split}_recall": m["recall"],
                        f"{split}_f1": m["f1"],
                        f"{split}_tp": m["tp"],
                        f"{split}_fp": m["fp"],
                        f"{split}_fn": m["fn"],
                        f"{split}_tn": m["tn"],
                    }
                )
            mlflow.log_metric("threshold", thr)
            mlflow.log_metric("repro_abs_diff_vs_phase4", drift)
            mlflow.set_tags(
                {
                    "python": platform.python_version(),
                    "sklearn": sklearn.__version__,
                    "phase": "6",
                    "data_note": "synthetic SAML-D; thresholds from validation",
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                ev.plot_curves({name: (yv, pv)}, tmp_dir / "curves_valid.png", "validation")
                ev.plot_curves({name: (yt, pt)}, tmp_dir / "curves_test.png", "test")
                ev.plot_confusions({name: mt}, tmp_dir / "confusion_test.png", "Test set")
                imp = _importance(fitted, name)
                pd.Series(imp).sort_values().plot.barh(
                    figsize=(7, 5), title=f"Top features - {name}"
                )
                plt.tight_layout()
                plt.savefig(tmp_dir / "feature_importance.png", dpi=130)
                plt.close()
                mlflow.log_dict(imp, "plots/feature_importance.json")
                for f in tmp_dir.glob("*.png"):
                    mlflow.log_artifact(str(f), "plots")
            for extra in (REP / "tuning_results.json", PROJECT_ROOT / "requirements.lock.txt"):
                mlflow.log_artifact(str(extra), "provenance")
            sig = infer_signature(raw_example, deploy.predict_proba(raw_example))
            mlflow.sklearn.log_model(
                deploy,
                name="model",
                signature=sig,
                input_example=raw_example,
                serialization_format="cloudpickle",
                pyfunc_predict_fn="predict_proba",
                pip_requirements=PIP_REQUIREMENTS,
                code_paths=[str(PROJECT_ROOT / "src" / "aml")],
            )
            run_ids[name] = run.info.run_id
        print(
            f"{name}: valid PR-AUC {rank_v['pr_auc']:.4f}, test PR-AUC {rank_t['pr_auc']:.4f}, "
            f"drift vs phase 4 {drift:.2e}"
        )

    # Compare runs and register the best by validation PR-AUC.
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string="tags.phase = '6'",
        order_by=["metrics.valid_pr_auc DESC"],
    )
    best = runs.iloc[0]
    best_id = best["run_id"]
    mv_reg = mlflow.register_model(f"runs:/{best_id}/model", REGISTERED_NAME)
    client.set_model_version_tag(REGISTERED_NAME, mv_reg.version, "stage", "Production")
    client.set_model_version_tag(REGISTERED_NAME, mv_reg.version, "alias", "champion")
    # Aliases may not be supported by every registry backend (tags above always work).
    try:
        client.set_registered_model_alias(REGISTERED_NAME, "champion", mv_reg.version)
        alias_ok = True
    except Exception as exc:
        alias_ok = False
        print(f"Alias not set ({type(exc).__name__}); using version tags only")
    model_uri = (
        f"models:/{REGISTERED_NAME}@champion"
        if alias_ok
        else f"models:/{REGISTERED_NAME}/{mv_reg.version}"
    )
    client.set_model_version_tag(
        REGISTERED_NAME, mv_reg.version, "threshold", str(best["metrics.threshold"])
    )
    client.set_model_version_tag(
        REGISTERED_NAME, mv_reg.version, "selection", "highest validation PR-AUC"
    )
    client.set_model_version_tag(REGISTERED_NAME, mv_reg.version, "data_sha256", data_hash)

    # Verify: the registered alias reproduces the saved Phase 4 artifact.
    loaded = mlflow.sklearn.load_model(model_uri)
    sample = pd.read_parquet(PROC / "saml_d.parquet", columns=REQUIRED_INPUT).iloc[-20_000:]
    sample_json_like = sample.assign(Timestamp=sample["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
    local = load_artifact(PROJECT_ROOT / "models" / "aml_pipeline.pkl")
    p_reg = loaded.predict_proba(sample_json_like)[:, 1]
    p_loc, _ = score(local, sample)
    parity = float(np.max(np.abs(p_reg - p_loc)))
    pyf = mlflow.pyfunc.load_model(model_uri)
    probe = raw_example.copy()
    p_pyf = pyf.predict(probe)[:, 1]
    p_ref = loaded.predict_proba(probe)[:, 1]
    probe_dt = probe.copy()
    probe_dt["Timestamp"] = pd.to_datetime(probe_dt["Timestamp"])
    try:
        dt_ok = bool(np.allclose(pyf.predict(probe_dt)[:, 1], p_ref))
    except Exception as exc:  # recorded, not fatal
        dt_ok = f"{type(exc).__name__}: {str(exc)[:120]}"
    summary = {
        "pyfunc_matches_sklearn_flavor": bool(np.allclose(p_pyf, p_ref)),
        "pyfunc_accepts_datetime_objects": dt_ok,
        "tracking_uri": tracking_uri,
        "run_ids": run_ids,
        "champion_run": best_id,
        "champion_model": best["params.model"],
        "registered_version": int(mv_reg.version),
        "alias": "champion",
        "alias_set": alias_ok,
        "registry_vs_pkl_max_abs_diff_20k_rows": parity,
        "comparison": runs[
            [
                "params.model",
                "metrics.valid_pr_auc",
                "metrics.test_pr_auc",
                "metrics.test_f1",
                "metrics.repro_abs_diff_vs_phase4",
            ]
        ]
        .round(6)
        .to_dict("records"),
    }
    (REP / "mlflow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparison"}, indent=1))


if __name__ == "__main__":
    main()
