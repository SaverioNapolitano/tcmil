"""Metric computation helpers for binary classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute all binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with all metric values.
    """
    # Ensure everything is finite before passing to sklearn
    if not np.all(np.isfinite(y_prob)):
        return {k: float("nan") for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Unweighted mean of per-class F1 — what the DAIC-WOZ text-only
        # literature reports as "F1"; our "f1" is positive-class only.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Micro-F1; for single-label binary this equals accuracy. Reported to
        # match Milintsevich et al. 2023, who give both micro- and macro-F1.
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    # ROC AUC and PR AUC require at least one positive and one negative sample
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix and return as a dictionary.

    Returns:
        Dictionary with keys: tn, fp, fn, tp.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1", pos_weight: float = 1.0
) -> float:
    """Find the threshold in [0.01, 0.99] that maximizes the given metric.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        metric: Either 'f1', 'balanced_accuracy', or 'loss'.
        pos_weight: Weight for the positive class (used only if metric='loss').

    Returns:
        Best threshold value.
    """
    best_t = 0.5
    if metric == "loss":
        best_score = float("inf")
    else:
        best_score = -1.0

    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        elif metric == "loss":
            # Weighted binary zero-one loss: pos_weight * FN + FP
            fn = np.sum((y_true == 1) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            score = pos_weight * fn + fp
        else:
            raise ValueError(f"Unknown tuning metric {metric}")

        if metric == "loss":
            if score < best_score:
                best_score = score
                best_t = t
        else:
            if score > best_score:
                best_score = score
                best_t = t

    return float(best_t)


def compute_attention_entropy(attention_weights: np.ndarray, eps: float = 1e-9) -> float:
    """Compute the entropy of an attention distribution.
    
    If the input is 2D, computes the mean entropy across the first dimension 
    (e.g., across multiple attention heads or rows).
    
    Args:
        attention_weights: Array of shape [Num_Utterances] or [N, Num_Utterances] summing to 1.
        eps: Small value to avoid log(0).
        
    Returns:
        Entropy value (or mean entropy).
    """
    if attention_weights.ndim > 1:
        # Compute entropy per row and return mean
        entropies = -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)
        return float(np.mean(entropies))
    
    return float(-np.sum(attention_weights * np.log(attention_weights + eps)))
