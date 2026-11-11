"""
src/evaluate.py
NetSentinel — Evaluation utilities and plot generation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

log = logging.getLogger(__name__)


# ── Core metrics ──────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximises Youden's J (sensitivity + specificity - 1)
    on the supplied data.  Falls back to 0.5 if the ROC curve is degenerate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    thresh = float(thresholds[best_idx])
    # Clamp to a sane range
    return float(np.clip(thresh, 0.05, 0.95))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    auto_threshold: bool = False,
) -> Dict[str, float]:
    """
    Return a dict of all evaluation metrics.

    Parameters
    ----------
    auto_threshold : bool
        If True, find the optimal Youden-J threshold from y_true/y_prob and
        use it for binary metrics (accuracy, precision, recall, f1).
        The chosen threshold is included in the result dict.
    """
    if auto_threshold:
        threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    res = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "pr_auc":    float(average_precision_score(y_true, y_prob)),
    }
    if auto_threshold:
        res["threshold"] = round(float(threshold), 4)
    return res


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


def print_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Attack"],
                                zero_division=0))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: str,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Normal", "Attack"]
    ax.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title=f"Confusion Matrix — {model_name}",
    )

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    path = Path(plots_dir) / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix: %s", path)


def plot_roc_curves(
    results: Dict[str, Dict],  # {name: {"y_true": ..., "y_prob": ...}}
    plots_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, d in results.items():
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        auc = roc_auc_score(d["y_true"], d["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(plots_dir) / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curves: %s", path)


def plot_pr_curves(
    results: Dict[str, Dict],
    plots_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, d in results.items():
        prec, rec, _ = precision_recall_curve(d["y_true"], d["y_prob"])
        pr_auc = average_precision_score(d["y_true"], d["y_prob"])
        ax.plot(rec, prec, label=f"{name} (AP={pr_auc:.4f})")

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = Path(plots_dir) / "pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved PR curves: %s", path)


def plot_training_history(history: dict, model_name: str, plots_dir: str) -> None:
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.get("loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set(title=f"{model_name} — Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="Train Acc")
    axes[1].plot(history.get("val_accuracy", []), label="Val Acc")
    axes[1].set(title=f"{model_name} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    path = Path(plots_dir) / f"history_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved training history: %s", path)


def comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build a DataFrame comparing all models."""
    rows = []
    for name, m in all_metrics.items():
        rows.append({"Model": name, **{k.upper(): round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows).set_index("Model")
    return df
