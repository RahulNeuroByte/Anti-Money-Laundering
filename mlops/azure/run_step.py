"""One pipeline step per invocation, for Azure ML command components (Python 3.11 job environment).

    python run_step.py prepare   --raw SAML-D.csv --prepared OUT
    python run_step.py features  --prepared IN --features OUT --neg-pool 200000
    python run_step.py imbalance --features IN --imbalance OUT
    python run_step.py tune      --features IN --imbalance IN --tuning OUT --cv-neg 120000
    python run_step.py final     --features IN --prepared IN --tuning IN \
                                 --model OUT --evaluation OUT

The existing aml modules are reused unchanged: a scratch project root (AML_PROJECT_ROOT) is laid
out under ./work, inputs are copied in, the module runs, outputs are copied out, and parameters,
metrics and PNG/JSON artifacts are logged with MLflow. Inside Azure ML the tracking URI and run are
preset by the service; elsewhere a local SQLite store is used (tests without Azure).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED_FILES = ("saml_d.parquet", "pool.parquet", "valid.parquet", "test.parquet",
                   "fb_static.pkl", "feature_columns.json")  # fmt: skip


# ---------------------------------------------------------------------------------- file helpers
def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _bring(src_dir: Path, names, dst_dir: Path) -> None:
    for name in names:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Expected input file not found: {src}")
        _copy(src, dst_dir / name)


def _send(src_dir: Path, names, dst_dir: Path) -> None:
    for name in names:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Step did not produce {src}")
        _copy(src, dst_dir / name)


def _send_figures(reports: Path, dst_dir: Path) -> list[Path]:
    out = []
    for fig in sorted((reports / "figures").glob("*.png")):
        _copy(fig, dst_dir / "figures" / fig.name)
        out.append(dst_dir / "figures" / fig.name)
    return out


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------- MLflow
def _tracking(work: Path):
    import mlflow

    in_azure = bool(os.environ.get("MLFLOW_RUN_ID"))
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(work / 'mlflow_local.db').as_posix()}")
        mlflow.set_experiment("aml-pipeline-local")
    return mlflow, in_azure


def _log(
    work: Path,
    step: str,
    params: dict,
    metrics: dict,
    files: list[tuple[Path, str]],
    series: dict[str, list[float]] | None = None,
):
    """Log to the job's MLflow run (Azure) or to a local run named after the step.

    `series` lists are logged as stepped metrics so the studio draws them as line charts.
    """
    mlflow, in_azure = _tracking(work)
    ctx = contextlib.nullcontext() if in_azure else mlflow.start_run(run_name=step)
    with ctx:
        mlflow.set_tag("pipeline_step", step)
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
        for i in range(0, len(numeric), 100):
            mlflow.log_metrics(dict(list(numeric.items())[i : i + 100]))
        for key, values in (series or {}).items():
            for i, v in enumerate(values):
                mlflow.log_metric(key, float(v), step=i)
        for path, artifact_dir in files:
            mlflow.log_artifact(str(path), artifact_dir)
    return numeric


# ------------------------------------------------------------------------------------- the steps
def step_prepare(a, work: Path, reports: Path, proc: Path) -> None:
    raw = work / "data" / "raw" / "SAML-D.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    src = Path(a.raw)
    if not src.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {src}")
    shutil.copy2(src, raw)
    from aml import eda, profile_data

    profile_data.main()
    eda.main()
    out = Path(a.prepared)
    _send(proc, ["saml_d.parquet"], out)
    names = ["dataset_profile.json", "dataset_preview.csv", "cleaning_log.json", "eda_summary.json"]
    _send(reports, names, out)
    figs = _send_figures(reports, out)
    prof, clean, eda_s = (_read(reports / n) for n in names[:1] + names[2:])
    metrics = {
        "rows": prof["shape"][0],
        "positives": prof["target"]["positives"],
        "positive_rate_percent": prof["target"]["positive_rate_percent"],
        "imbalance_ratio_neg_to_pos": prof["target"]["imbalance_ratio_neg_to_pos"],
        "duplicates_removed": clean["duplicates_removed"],
        "amount_outliers_count": clean["amount_outliers"]["count_above_fence"],
        "unique_senders": prof["unique_senders"],
        "unique_receivers": prof["unique_receivers"],
    }
    for split, v in eda_s["split"].items():
        metrics[f"{split}_rows"], metrics[f"{split}_positives"] = v["rows"], v["positives"]
    for feat, auc in eda_s["univariate_auc_train_sample"].items():
        metrics[f"univariate_auc.{feat}"] = auc
    files = [(out / n, "reports") for n in names] + [(f, "plots") for f in figs]
    _log(work, "prepare", {"target": prof["target"]["name"]}, metrics, files)


def step_features(a, work: Path, reports: Path, proc: Path) -> None:
    _bring(Path(a.prepared), ["saml_d.parquet"], proc)
    from aml import train

    train.stage_features(a.neg_pool)
    out = Path(a.features)
    _send(proc, [n for n in PROCESSED_FILES if n != "saml_d.parquet"], out)
    meta = _read(proc / "feature_columns.json")
    import pyarrow.parquet as pq

    metrics = {f"{n}_rows": pq.ParquetFile(proc / f"{n}.parquet").metadata.num_rows
               for n in ("pool", "valid", "test")}  # fmt: skip
    metrics["pool_positives"], metrics["pool_negatives"] = (
        meta["pool_positives"],
        meta["pool_negatives"],
    )
    metrics["n_features"] = len(meta["columns"])
    params = {k: meta[k] for k in ("window_days", "block_days", "warmup_end", "static_profile_end")}
    _log(work, "features", {**params, "neg_pool": a.neg_pool}, metrics,
         [(out / "feature_columns.json", "reports")])  # fmt: skip


def _features_in(a, proc: Path) -> None:
    _bring(Path(a.features), [n for n in PROCESSED_FILES if n != "saml_d.parquet"], proc)


def step_imbalance(a, work: Path, reports: Path, proc: Path) -> None:
    _features_in(a, proc)
    from aml import train

    train.stage_imbalance()
    out = Path(a.imbalance)
    _send(reports, ["imbalance_results.json", "imbalance_results.csv", "best_strategy.json"], out)
    figs = _send_figures(reports, out)
    metrics = {}
    for r in _read(reports / "imbalance_results.json"):
        for k in ("pr_auc", "roc_auc", "best_f1", "precision", "recall"):
            metrics[f"{k}.{r['model']}.{r['strategy']}"] = r[k]
    files = [(out / n, "reports") for n in ("imbalance_results.json", "imbalance_results.csv")]
    _log(work, "imbalance", {"strategies": "none,class_weight,random_over,random_under,smotenc"},
         metrics, files + [(f, "plots") for f in figs])  # fmt: skip


def step_tune(a, work: Path, reports: Path, proc: Path) -> None:
    _features_in(a, proc)
    _bring(Path(a.imbalance), ["best_strategy.json"], reports)
    from aml import train

    train.stage_tune(a.cv_neg)
    out = Path(a.tuning)
    _send(reports, ["tuning_results.json"], out)
    tuned = _read(reports / "tuning_results.json")
    metrics = {f"cv_average_precision.{m}": t["cv_average_precision"] for m, t in tuned.items()}
    params = {f"{m}.{k}": v for m, t in tuned.items() for k, v in {**t["params"],
              "strategy": t["strategy"]}.items()}  # fmt: skip
    _log(
        work,
        "tune",
        {**params, "cv_neg": a.cv_neg},
        metrics,
        [(out / "tuning_results.json", "reports")],
    )


def step_final(a, work: Path, reports: Path, proc: Path) -> None:
    _features_in(a, proc)
    _bring(Path(a.prepared), ["saml_d.parquet"], proc)
    _bring(Path(a.tuning), ["tuning_results.json"], reports)
    from aml import train, verify

    train.stage_final()
    verify.validate_artifact()
    models = work / "models"
    model_out, eval_out = Path(a.model), Path(a.evaluation)
    _send(models, ["aml_pipeline.pkl", "model_meta.json"], model_out)
    eval_names = ["model_results.json", "artifact_validation.json", "threshold_sweep.json"]
    _send(reports, eval_names, eval_out)
    figs = _send_figures(reports, eval_out)
    res, sweep = _read(reports / "model_results.json"), _read(reports / "threshold_sweep.json")
    metrics = {}
    for m in ("logreg", "tree", "forest"):
        for split in ("valid", "test"):
            r = res[m][split]
            at = r["at_best_f1"]
            metrics[f"{split}.pr_auc.{m}"], metrics[f"{split}.roc_auc.{m}"] = (
                r["pr_auc"],
                r["roc_auc"],
            )
            for k in ("precision", "recall", "f1", "accuracy", "threshold", "tp", "fp", "fn", "tn"):
                metrics[f"{split}.{k}.{m}"] = at[k]
    series = {}
    for m, by_split in sweep.items():
        for split, sw in by_split.items():
            for k in ("precision", "recall", "f1"):
                series[f"sweep.{split}.{k}.{m}"] = sw[k]
    series["sweep.threshold_grid"] = sweep["forest"]["valid"]["threshold"]
    files = [(eval_out / n, "reports") for n in eval_names] + [(f, "plots") for f in figs]
    files += [(model_out / "aml_pipeline.pkl", "model"), (model_out / "model_meta.json", "model")]
    params = {"selected_model": res["selected"],
              "threshold": res[res["selected"]]["valid"]["at_best_f1"]["threshold"]}  # fmt: skip
    _log(work, "final", params, metrics, files, series)


STEPS = {"prepare": step_prepare, "features": step_features, "imbalance": step_imbalance,
         "tune": step_tune, "final": step_final}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    for name in ("raw", "prepared", "features", "imbalance", "tuning", "model", "evaluation"):
        ap.add_argument(f"--{name}")
    ap.add_argument("--neg-pool", type=int, default=200_000)
    ap.add_argument("--cv-neg", type=int, default=120_000)
    ap.add_argument("--work", default=str(Path.cwd() / "work"))
    a = ap.parse_args(argv)

    work = Path(a.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    proc, reports = work / "data" / "processed", work / "reports"
    for d in (proc, reports / "figures", work / "models", work / "configs"):
        d.mkdir(parents=True, exist_ok=True)
    cfg = HERE / "configs" / "config.yaml"
    shutil.copy2(cfg if cfg.is_file() else HERE.parents[2] / "configs" / "config.yaml",
                 work / "configs" / "config.yaml")  # fmt: skip
    os.environ["AML_PROJECT_ROOT"] = str(work)
    sys.path.insert(0, str(HERE))
    STEPS[a.step](a, work, reports, proc)
    print(f"Step '{a.step}' finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
