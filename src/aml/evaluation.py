"""Metrics, threshold selection and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "threshold": float(thr), "accuracy": (tp + tn) / len(y), "precision": prec,
        "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support_pos": int(np.sum(y == 1)), "support_neg": int(np.sum(y == 0)),
        "alerts": tp + fp,
    }  # fmt: skip


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold that still reaches `target` recall."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def plot_curves(scores: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (y, p) in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        ax[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {title}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall - {title}")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusions(mets: dict[str, dict[str, float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, len(mets), figsize=(4.2 * len(mets), 3.8))
    ax = np.atleast_1d(ax)
    for a, (name, m) in zip(ax, mets.items(), strict=True):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(np.log10(cm + 1), cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9)
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Laundering"],
              yticklabels=["Normal", "Laundering"], xlabel="Predicted", ylabel="Actual",
              title=f"{name}\n(thr {m['threshold']:.3f})")  # fmt: skip
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def threshold_sweep(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> dict[str, list[float]]:
    """Precision, recall and F1 over a grid of decision thresholds."""
    out: dict[str, list[float]] = {"threshold": [], "precision": [], "recall": [], "f1": []}
    for t in grid:
        m = threshold_metrics(y, p, float(t))
        out["threshold"].append(float(t))
        for k in ("precision", "recall", "f1"):
            out[k].append(float(m[k]))
    return out


def plot_sweeps(
    valid: dict[str, dict[str, list[float]]],
    test_selected: dict[str, list[float]],
    selected: str,
    chosen_threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, sw in valid.items():
        ax[0].plot(sw["threshold"], sw["f1"], label=name)
    ax[0].axvline(chosen_threshold, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="Decision threshold", ylabel="F1", title="F1 vs threshold (validation)")
    ax[0].legend(fontsize=8)
    ax[1].plot(test_selected["threshold"], test_selected["precision"], label="precision")
    ax[1].plot(test_selected["threshold"], test_selected["recall"], label="recall")
    ax[1].plot(test_selected["threshold"], test_selected["f1"], label="F1")
    ax[1].axvline(chosen_threshold, color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="Decision threshold", title=f"Test metrics vs threshold ({selected})")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
