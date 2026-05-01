"""
Evaluation metrics and visualisation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

ELEV_BINS   = [(50, 65), (65, 75), (75, 85), (85, 91)]
ELEV_LABELS = ["50-65°", "65-75°", "75-85°", "85-90°"]

MODEL_COLORS = {
    "Threshold": "#e74c3c",
    "LogReg":    "#f39c12",
    "MLP":       "#27ae60",
    "GP":        "#2980b9",
}


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bins, total, n = np.linspace(0, 1, n_bins + 1), 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if not m.any():
            continue
        total += m.sum() / n * abs(y_true[m].mean() - y_prob[m].mean())
    return total


def compute_all(y_true, y_prob, name: str = "") -> dict:
    return {
        "name":  name,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "ECE":   ece(y_true, y_prob),
    }


def stratified(y_true, y_prob, elev,
               bins=ELEV_BINS, labels=ELEV_LABELS) -> dict:
    out = {}
    for (lo, hi), lbl in zip(bins, labels):
        m = (elev >= lo) & (elev < hi)
        if m.sum() < 10 or y_true[m].sum() == 0:
            continue
        out[lbl] = {
            "n":     int(m.sum()),
            "AUROC": roc_auc_score(y_true[m], y_prob[m]),
            "AUPRC": average_precision_score(y_true[m], y_prob[m]),
            "ECE":   ece(y_true[m], y_prob[m]),
        }
    return out


def reliability_diagram(ax, y_true, y_prob, n_bins=15, label="", color="#2980b9"):
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, weights = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if not m.any():
            continue
        accs.append(y_true[m].mean())
        confs.append(y_prob[m].mean())
        weights.append(m.sum())
    if not accs:
        return
    accs, confs = np.array(accs), np.array(confs)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.scatter(confs, accs, s=np.array(weights) / max(weights) * 120,
               color=color, zorder=3, label=label)
    ax.plot(confs, accs, color=color, alpha=0.5, lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_roc_pr(results: list) -> plt.Figure:
    """results = list of dicts with name, y_true, y_prob."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in results:
        c = MODEL_COLORS.get(r["name"], "#333333")
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        p, rec, _   = precision_recall_curve(r["y_true"], r["y_prob"])
        auroc = roc_auc_score(r["y_true"], r["y_prob"])
        auprc = average_precision_score(r["y_true"], r["y_prob"])
        axes[0].plot(fpr, tpr, color=c, lw=1.8,
                     label=f"{r['name']} ({auroc:.3f})")
        axes[1].plot(rec, p, color=c, lw=1.8,
                     label=f"{r['name']} ({auprc:.3f})")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
