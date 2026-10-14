"""Phase 3 runner: cleaning, split audit, leakage audit, EDA figures, feature signal check."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from aml.cleaning import clean, save_report  # noqa: E402
from aml.config import PROJECT_ROOT, load_config  # noqa: E402
from aml.data import load_dataset  # noqa: E402
from aml.features import CATEGORICAL, FeatureBuilder  # noqa: E402
from aml.logging_utils import get_logger  # noqa: E402
from aml.splitting import split_summary, time_split  # noqa: E402

log = get_logger("aml.eda")
FIG = PROJECT_ROOT / "reports" / "figures"
SEED = 42


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def _rate_bar(train: pd.DataFrame, col: str, target: str, name: str, title: str) -> None:
    r = train.groupby(col, observed=True)[target].mean().sort_values() * 100
    r.plot.barh(figsize=(7, max(3, 0.28 * len(r))), title=title)
    plt.xlabel("Positive rate (%)")
    _save(name)


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    FIG.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    df, clean_report = clean(load_dataset(), target)
    save_report(clean_report, PROJECT_ROOT / "reports" / "cleaning_log.json")

    # Leakage audit on the label-bearing column.
    ct = pd.crosstab(df["Laundering_type"], df[target])
    mixed = ct[(ct[0] > 0) & (ct[1] > 0)]
    summary["leakage_audit"] = {
        "laundering_type_values": int(len(ct)),
        "values_appearing_in_both_classes": int(len(mixed)),
        "conclusion": "Laundering_type fully determines the target; excluded from features",
    }

    sp = cfg["split"]
    train, valid, test = time_split(df, sp["train_frac"], sp["valid_frac"])
    summary["split"] = split_summary({"train": train, "valid": valid, "test": test}, target)
    del df, valid, test  # EDA below uses the training window only

    # Descriptive figures (training window only; test stays untouched).
    m = train.groupby(train["Timestamp"].dt.to_period("M"))[target].agg(["size", "sum"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    m["size"].plot.bar(ax=ax[0], title="Transactions per month (train)")
    m["sum"].plot.bar(ax=ax[1], color="firebrick", title="Laundering per month (train)")
    _save("01_monthly_volume.png")

    la = np.log10(train["Amount"].astype("float64"))
    plt.figure(figsize=(7, 4))
    bins = np.linspace(la.min(), la.max(), 70)
    plt.hist(la[train[target] == 0], bins=bins, density=True, alpha=0.6, label="Normal")
    plt.hist(la[train[target] == 1], bins=bins, density=True, alpha=0.6, label="Laundering")
    plt.xlabel("log10(Amount)")
    plt.legend()
    plt.title("Amount distribution by class (train)")
    _save("02_amount_by_class.png")

    _rate_bar(train, "Payment_type", target, "03_rate_payment_type.png", "Rate by payment type")
    _rate_bar(
        train,
        "Sender_bank_location",
        target,
        "04_rate_sender_location.png",
        "Laundering rate by sender location",
    )
    _rate_bar(
        train,
        "Received_currency",
        target,
        "05_rate_received_currency.png",
        "Laundering rate by received currency",
    )
    plt.figure(figsize=(8, 3.4))
    (train.groupby(train["Timestamp"].dt.hour)[target].mean() * 100).plot(marker="o")
    plt.ylabel("Positive rate (%)")
    plt.title("Laundering rate by hour of day (train)")
    _save("06_rate_by_hour.png")

    lt = train.loc[train[target] == 1, "Laundering_type"].astype(str).value_counts()
    lt.sort_values().plot.barh(figsize=(7, 5), title="Typologies in train (analysis only)")
    _save("07_typology_counts.png")

    # Feature signal on a training sample: all positives plus 500k random negatives.
    rng = np.random.default_rng(SEED)
    pos = train.index[train[target] == 1]
    neg = rng.choice(train.index[train[target] == 0], size=500_000, replace=False)
    fb = FeatureBuilder().fit(train)
    sample = train.loc[np.sort(np.concatenate([pos, neg]))]
    feats = fb.transform(sample)
    y = sample[target].to_numpy()
    num = feats.drop(columns=CATEGORICAL)
    auc = {c: float(roc_auc_score(y, num[c])) for c in num.columns if num[c].nunique() > 1}
    strength = pd.Series(auc).sub(0.5).abs().sort_values(ascending=False)
    summary["univariate_auc_train_sample"] = {k: round(auc[k], 4) for k in strength.index}

    corr = num.assign(target=y).corr(method="spearman")
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.columns, fontsize=6)
    plt.colorbar()
    plt.title("Spearman correlation (train sample, engineered features)")
    _save("08_correlation.png")
    summary["top_target_correlations"] = {
        k: round(float(v), 4) for k, v in corr["target"].drop("target").abs().nlargest(10).items()
    }
    summary["feature_columns"] = list(feats.columns)
    summary["n_features"] = int(feats.shape[1])

    (PROJECT_ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("EDA complete; figures in %s", FIG)


if __name__ == "__main__":
    main()
