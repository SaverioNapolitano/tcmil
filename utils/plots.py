"""Plotting helpers for binary classification experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_dir: Path,
) -> None:
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_metric_curves(
    history: dict[str, list[float]],
    metric_names: list[str],
    output_dir: Path,
) -> None:
    """Plot metric curves over epochs for train and val."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in metric_names:
        for split in ["train", "val"]:
            key = f"{split}_{name}"
            if key in history:
                epochs = range(1, len(history[key]) + 1)
                ax.plot(epochs, history[key], label=f"{split} {name}", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Metrics Over Epochs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metric_curves.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, split_name: str, output_dir: Path
) -> None:
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=split_name)
    ax.set_title(f"ROC Curve — {split_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_curve_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, split_name: str, output_dir: Path
) -> None:
    """Plot and save Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=split_name)
    ax.set_title(f"Precision-Recall Curve — {split_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"pr_curve_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, split_name: str, output_dir: Path
) -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Depressed", "Depressed"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {split_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_probability_histogram(
    y_true: np.ndarray, y_prob: np.ndarray, split_name: str, output_dir: Path
) -> None:
    """Plot histogram of predicted probabilities, colored by true class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        y_prob[y_true == 0], bins=20, alpha=0.6, label="Not Depressed (true=0)",
        color="steelblue", edgecolor="white",
    )
    ax.hist(
        y_prob[y_true == 1], bins=20, alpha=0.6, label="Depressed (true=1)",
        color="salmon", edgecolor="white",
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted Probability Distribution — {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"prob_histogram_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_utterance_distribution(
    utterance_counts: dict[str, list[int]], output_dir: Path
) -> None:
    """Plot distribution of number of utterances per interview across splits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for split_name, counts in utterance_counts.items():
        ax.hist(counts, bins=30, alpha=0.5, label=split_name, edgecolor="white")
    ax.set_xlabel("Number of Utterances per Interview")
    ax.set_ylabel("Count")
    ax.set_title("Utterance Count Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "utterance_distribution.png", dpi=150)
    plt.close(fig)


def plot_prob_vs_bag_size(
    bag_sizes: np.ndarray, y_prob: np.ndarray, split_name: str, output_dir: Path
) -> None:
    """Plot scatter plot of predicted probabilities vs bag size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bag_sizes, y_prob, alpha=0.6, edgecolors="w", s=40)
    ax.set_xlabel("Bag Size (Number of Utterances)")
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"Predicted Probability vs Bag Size — {split_name}")
    
    # Add a horizontal line at 0.5 decision threshold
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Decision Threshold (0.5)")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_dir / f"prob_vs_bag_size_{split_name}.png", dpi=150)
    plt.close(fig)

