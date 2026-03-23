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
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> float:
    """Find the threshold in [0.01, 0.99] that maximizes the given metric.
    
    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        metric: Either 'f1' or 'balanced_accuracy'.
        
    Returns:
        Best threshold value.
    """
    best_t = 0.5
    best_score = -1.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            raise ValueError(f"Unknown tuning metric {metric}")
            
        if score > best_score:
            best_score = score
            best_t = t
            
    return float(best_t)


def compute_attention_entropy(attention_weights: np.ndarray, eps: float = 1e-9) -> float:
    """Compute the entropy of an attention distribution.
    
    Args:
        attention_weights: Array of shape [Num_Utterances] summing to 1.
        eps: Small value to avoid log(0).
        
    Returns:
        Entropy value.
    """
    return float(-np.sum(attention_weights * np.log(attention_weights + eps)))
