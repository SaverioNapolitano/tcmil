"""
Utility functions for baseline evaluation: metrics computation and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all required classification metrics.

    Args:
        y_true: ground-truth binary labels (0/1).
        y_pred: predicted binary labels (0/1).
        y_prob: predicted probability of the positive class (optional).

    Returns:
        dict with metric names as keys.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # ROC AUC needs both classes present in y_true
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


# --------------- Plotting helpers ---------------


def plot_class_distribution(splits_dict, path):
    """Bar chart of class distribution across splits.

    Args:
        splits_dict: {"train": df, "val": df, "test": df} with a 'label' column.
        path: file path to save the figure.
    """
    records = []
    for split_name, df in splits_dict.items():
        counts = df["label"].value_counts().sort_index()
        for label_val, count in counts.items():
            records.append({"split": split_name, "label": int(label_val), "count": count})

    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(6, 4))
    splits = list(splits_dict.keys())
    x = np.arange(len(splits))
    width = 0.35

    for i, label_val in enumerate([0, 1]):
        subset = plot_df[plot_df["label"] == label_val]
        # Align order with splits list
        counts = [subset[subset["split"] == s]["count"].values[0] if len(subset[subset["split"] == s]) else 0 for s in splits]
        ax.bar(x + i * width, counts, width, label=f"Label {label_val}")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, title, path):
    """Heatmap of a 2×2 confusion matrix.

    Args:
        cm: 2×2 list or array [[TN, FP], [FN, TP]].
        title: plot title.
        path: file path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_boxplot(seed_metrics_df, metric_cols, title, path):
    """Boxplot of metrics across seeds.

    Args:
        seed_metrics_df: DataFrame with one row per seed, columns include metric_cols.
        metric_cols: list of metric column names to plot.
        title: plot title.
        path: file path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    data_to_plot = seed_metrics_df[metric_cols]
    data_to_plot.boxplot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
