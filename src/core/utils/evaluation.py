"""Generic Monte Carlo Cross Validation framework.

Supports repeated stratified splits at the subject/interview level,
with multiple random seeds per split, for stable and trustworthy evaluation.
"""

from typing import Callable, Any
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, LeaveOneGroupOut

from src.core.utils.stats import compute_aggregate_metrics, compute_bootstrap_metrics
from src.core.utils.metrics import compute_metrics


def run_monte_carlo_cv(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, float]],
    n_splits: int = 5,
    n_seeds_per_split: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Run Repeated Stratified Monte Carlo Cross Validation at the Subject Level.

    For each split, unique subjects are partitioned into a training pool and a hold-out test set.
    This strictly enforces speaker independence, even if a speaker has multiple interviews.

    Args:
        interviews: List of all interviews (train + dev + test combined).
        train_eval_fn: Callback function `fn(train_pool, test_set, seed) -> metrics_dict`.
        n_splits: Number of random monte carlo splits to generate.
        n_seeds_per_split: Number of independent training runs per split.
        test_size: Proportion of unique subjects to hold out for the test set.
        random_state: Base seed for the random weight generation.

    Returns:
        A tuple:
        - Aggregated metrics (mean, std, 95% CI)
        - List of all raw metric dictionaries from every run
    """
    # 1. Identify unique subjects and their labels for stratification
    subject_map = {}
    for iv in interviews:
        sid = iv["interview_id"]
        label = iv["label"]
        if sid not in subject_map:
            subject_map[sid] = label
    
    unique_sids = sorted(list(subject_map.keys()))
    unique_labels = [subject_map[sid] for sid in unique_sids]

    # 2. Setup StratifiedShuffleSplit on unique subjects
    cv = StratifiedShuffleSplit(
        n_splits=n_splits, 
        test_size=test_size, 
        random_state=random_state
    )

    all_raw_metrics = []
    
    print(f"\n={ '='*70 }=")
    print(f" Starting Subject-Level Monte Carlo CV")
    print(f" Unique Subjects: {len(unique_sids)}")
    print(f" Splits: {n_splits}, Seeds/Split: {n_seeds_per_split}, Test size: {test_size:.1%}")
    print(f" Total training runs: {n_splits * n_seeds_per_split}")
    print(f"={ '='*70 }=")

    # 3. Execute splits
    for split_idx, (train_subj_idx, test_subj_idx) in enumerate(cv.split(np.zeros(len(unique_labels)), unique_labels), 1):
        train_sids = set(unique_sids[i] for i in train_subj_idx)
        test_sids = set(unique_sids[i] for i in test_subj_idx)

        # Map back to full records
        train_pool = [iv for iv in interviews if iv["interview_id"] in train_sids]
        test_set = [iv for iv in interviews if iv["interview_id"] in test_sids]

        print(f"\n─── Split {split_idx}/{n_splits} ───")
        print(f"Train subjects: {len(train_sids)}, Records: {len(train_pool)}")
        print(f"Test subjects: {len(test_sids)}, Records: {len(test_set)}")

        for seed_idx in range(n_seeds_per_split):
            run_seed = random_state + (split_idx * 100) + seed_idx
            print(f"\n  [Split {split_idx}, Run {seed_idx+1}/{n_seeds_per_split}] Seed = {run_seed}")
            
            metrics = train_eval_fn(train_pool, test_set, run_seed)
            
            # Store run metadata
            metrics["_split_idx"] = split_idx
            metrics["_seed_idx"] = seed_idx
            metrics["_run_seed"] = run_seed
            all_raw_metrics.append(metrics)
            
            # Print a quick summary of this run's primary metrics
            f1 = metrics.get('f1', 0.0)
            bacc = metrics.get('balanced_accuracy', 0.0)
            print(f"  -> Result: F1={f1:.4f}, B.Acc={bacc:.4f}")

    # 4. Aggregation
    print("\nAggregating results across all runs...")
    agg_metrics = compute_aggregate_metrics(all_raw_metrics)
    
    return agg_metrics, all_raw_metrics


def run_stratified_group_k_fold(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, float]],
    n_folds: int = 5,
    n_seeds_per_fold: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Run Deterministic Stratified Group K-Fold Cross Validation.

    Each subject (interview_id) is assigned to exactly one fold.
    The folds are balanced by clinical label (stratified).

    Args:
        interviews: List of all interviews.
        train_eval_fn: Callback function `fn(train_pool, test_set, seed)`.
        n_folds: Number of folds (e.g., 5-fold or 10-fold).
        n_seeds_per_fold: Number of seeds to run per fold for stability.
        random_state: Base seed for reproducibility.

    Returns:
        A tuple:
        - Aggregated metrics (mean, std, 95% CI)
        - List of all raw metric dictionaries
    """
    labels = [iv["label"] for iv in interviews]
    groups = [iv["interview_id"] for iv in interviews]
    
    cv = StratifiedGroupKFold(
        n_splits=n_folds, 
        shuffle=True, 
        random_state=random_state
    )

    all_raw_metrics = []
    
    print(f"\n={ '='*70 }=")
    print(f" Starting Stratified Group K-Fold CV")
    print(f" Folds: {n_folds}, Seeds/Fold: {n_seeds_per_fold}")
    print(f" Total training runs: {n_folds * n_seeds_per_fold}")
    print(f"={ '='*70 }=")

    # Convert to numpy for indexing
    ivs_np = np.array(interviews)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(ivs_np, labels, groups), 1):
        train_pool = ivs_np[train_idx].tolist()
        test_set = ivs_np[test_idx].tolist()

        n_train_sub = len(set(iv["interview_id"] for iv in train_pool))
        n_test_sub = len(set(iv["interview_id"] for iv in test_set))

        print(f"\n─── Fold {fold_idx}/{n_folds} ───")
        print(f"Train subjects: {n_train_sub}, Records: {len(train_pool)}")
        print(f"Test subjects: {n_test_sub}, Records: {len(test_set)}")

        for seed_idx in range(n_seeds_per_fold):
            run_seed = random_state + (fold_idx * 100) + seed_idx
            print(f"\n  [Fold {fold_idx}, Run {seed_idx+1}/{n_seeds_per_fold}] Seed = {run_seed}")
            
            metrics = train_eval_fn(train_pool, test_set, run_seed)
            
            # Store run metadata
            metrics["_fold_idx"] = fold_idx
            metrics["_seed_idx"] = seed_idx
            metrics["_run_seed"] = run_seed
            all_raw_metrics.append(metrics)

    # Aggregation
    agg_metrics = compute_aggregate_metrics(all_raw_metrics)
    return agg_metrics, all_raw_metrics


def _seed_ensemble_metrics(
    group_seed_predictions: dict[int, list[dict]],
    group_key: str,
    ensemble_threshold_fn: Callable | None = None,
) -> list[dict]:
    """Average probabilities across seeds within each group (fold/split) and
    recompute metrics.

    ensemble_threshold_fn(group_idx, val_y_true, avg_val_probs, avg_test_probs)
    overrides the default loss-based inner-val threshold; val args are None
    when the callback did not return val predictions.
    """
    ensemble_metrics = []
    for group_idx in sorted(group_seed_predictions.keys()):
        seed_preds = group_seed_predictions[group_idx]
        if not seed_preds:
            continue

        all_probs = np.stack([sp["probability"] for sp in seed_preds])
        avg_probs = np.mean(all_probs, axis=0)
        y_true = seed_preds[0]["true_label"]  # Same for all seeds in this group

        avg_val_probs, val_y_true = None, None
        if "val_probability" in seed_preds[0] and "val_true_label" in seed_preds[0]:
            avg_val_probs = np.mean(
                np.stack([sp["val_probability"] for sp in seed_preds]), axis=0)
            val_y_true = seed_preds[0]["val_true_label"]

        if ensemble_threshold_fn is not None:
            best_t = float(ensemble_threshold_fn(
                group_idx, val_y_true, avg_val_probs, avg_probs))
        elif avg_val_probs is not None:
            num_pos = max(1, np.sum(val_y_true == 1))
            neg_to_pos = np.sum(val_y_true == 0) / num_pos
            from src.core.utils.metrics import find_best_threshold
            best_t = find_best_threshold(
                val_y_true, avg_val_probs, metric="loss", pos_weight=neg_to_pos)
        else:
            best_t = 0.5

        y_pred = (avg_probs >= best_t).astype(int)
        group_metrics = compute_metrics(y_true, y_pred, avg_probs)
        group_metrics[group_key] = group_idx
        group_metrics["_ensemble_threshold"] = float(best_t)
        ensemble_metrics.append(group_metrics)

        print(f"\n  {group_key.strip('_')} {group_idx} ensemble (t={best_t:.2f}): "
              f"ROC-AUC={group_metrics['roc_auc']:.4f}, "
              f"F1={group_metrics['f1']:.4f}")
    return ensemble_metrics


def run_monte_carlo_cv_ensemble(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, float]],
    n_splits: int = 5,
    n_seeds_per_split: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    ensemble_threshold_fn: Callable | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict], dict[str, dict[str, float]]]:
    """Monte Carlo CV with per-split seed-ensemble aggregation.

    Identical splits to run_monte_carlo_cv, but additionally averages the
    predicted probabilities across seeds within each split and recomputes
    metrics — the MC counterpart of run_stratified_group_k_fold_ensemble.
    """
    subject_map = {}
    for iv in interviews:
        if iv["interview_id"] not in subject_map:
            subject_map[iv["interview_id"]] = iv["label"]
    unique_sids = sorted(subject_map.keys())
    unique_labels = [subject_map[sid] for sid in unique_sids]

    cv = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)

    all_raw_metrics = []
    split_seed_predictions: dict[int, list[dict]] = {}

    print(f"\n={'='*70}=")
    print(f" Starting Subject-Level Monte Carlo CV (Seed-Ensemble)")
    print(f" Unique Subjects: {len(unique_sids)}")
    print(f" Splits: {n_splits}, Seeds/Split: {n_seeds_per_split}, Test size: {test_size:.1%}")
    print(f" Total training runs: {n_splits * n_seeds_per_split}")
    print(f"={'='*70}=")

    for split_idx, (train_subj_idx, test_subj_idx) in enumerate(
            cv.split(np.zeros(len(unique_labels)), unique_labels), 1):
        train_sids = set(unique_sids[i] for i in train_subj_idx)
        test_sids = set(unique_sids[i] for i in test_subj_idx)
        train_pool = [iv for iv in interviews if iv["interview_id"] in train_sids]
        test_set = [iv for iv in interviews if iv["interview_id"] in test_sids]

        print(f"\n─── Split {split_idx}/{n_splits} ───")
        print(f"Train subjects: {len(train_sids)}, Test subjects: {len(test_sids)}")

        split_seed_predictions[split_idx] = []
        for seed_idx in range(n_seeds_per_split):
            run_seed = random_state + (split_idx * 100) + seed_idx
            print(f"\n  [Split {split_idx}, Run {seed_idx+1}/{n_seeds_per_split}] Seed = {run_seed}")

            metrics = train_eval_fn(train_pool, test_set, run_seed)
            metrics["_split_idx"] = split_idx
            metrics["_seed_idx"] = seed_idx
            metrics["_run_seed"] = run_seed
            all_raw_metrics.append(metrics)

            pred_dict = {}
            if "probability" in metrics and "true_label" in metrics:
                pred_dict["probability"] = np.array(metrics["probability"])
                pred_dict["true_label"] = np.array(metrics["true_label"])
            if "val_probability" in metrics and "val_true_label" in metrics:
                pred_dict["val_probability"] = np.array(metrics["val_probability"])
                pred_dict["val_true_label"] = np.array(metrics["val_true_label"])
            if pred_dict:
                split_seed_predictions[split_idx].append(pred_dict)

    per_run_agg = compute_aggregate_metrics(all_raw_metrics)
    split_ensemble_metrics = _seed_ensemble_metrics(
        split_seed_predictions, "_split_idx", ensemble_threshold_fn)
    split_ensemble_agg = compute_aggregate_metrics(split_ensemble_metrics)
    return per_run_agg, all_raw_metrics, split_ensemble_agg


def run_stratified_group_k_fold_ensemble(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, float]],
    n_folds: int = 5,
    n_seeds_per_fold: int = 5,
    random_state: int = 42,
    ensemble_threshold_fn: Callable | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict], dict[str, dict[str, float]]]:
    """Stratified Group K-Fold with per-fold seed-ensemble aggregation.

    Runs the standard K-Fold CV, but additionally computes fold-level
    ensemble metrics by averaging predicted probabilities across seeds
    within each fold before re-computing metrics.

    This produces more stable fold-level estimates (variance reduced by
    √n_seeds) and tighter confidence intervals.

    Args:
        interviews: List of all interviews.
        train_eval_fn: Callback `fn(train_pool, test_set, seed) -> metrics_dict`.
                       Must return 'probability' (list[float]) and 'true_label'
                       (list[int]) in addition to standard metrics.
        n_folds: Number of folds.
        n_seeds_per_fold: Number of seeds per fold.
        random_state: Base seed.

    Returns:
        A tuple of:
        - per_run_agg: Aggregated metrics across all individual runs (standard)
        - all_raw_metrics: List of all per-run metric dicts
        - fold_ensemble_agg: Aggregated metrics from fold-level ensembles
    """
    labels = [iv["label"] for iv in interviews]
    groups = [iv["interview_id"] for iv in interviews]

    cv = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    all_raw_metrics = []
    # Collect predictions per fold for ensemble
    fold_seed_predictions: dict[int, list[dict]] = {}

    print(f"\n={'='*70}=")
    print(f" Starting Stratified Group K-Fold CV (Seed-Ensemble)")
    print(f" Folds: {n_folds}, Seeds/Fold: {n_seeds_per_fold}")
    print(f" Total training runs: {n_folds * n_seeds_per_fold}")
    print(f"={'='*70}=")

    ivs_np = np.array(interviews)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(ivs_np, labels, groups), 1):
        train_pool = ivs_np[train_idx].tolist()
        test_set = ivs_np[test_idx].tolist()

        n_train_sub = len(set(iv["interview_id"] for iv in train_pool))
        n_test_sub = len(set(iv["interview_id"] for iv in test_set))

        print(f"\n─── Fold {fold_idx}/{n_folds} ───")
        print(f"Train subjects: {n_train_sub}, Records: {len(train_pool)}")
        print(f"Test subjects: {n_test_sub}, Records: {len(test_set)}")

        fold_seed_predictions[fold_idx] = []

        for seed_idx in range(n_seeds_per_fold):
            run_seed = random_state + (fold_idx * 100) + seed_idx
            print(f"\n  [Fold {fold_idx}, Run {seed_idx+1}/{n_seeds_per_fold}] Seed = {run_seed}")

            metrics = train_eval_fn(train_pool, test_set, run_seed)

            metrics["_fold_idx"] = fold_idx
            metrics["_seed_idx"] = seed_idx
            metrics["_run_seed"] = run_seed
            all_raw_metrics.append(metrics)

            # Store predictions for fold-ensemble
            pred_dict = {}
            if "probability" in metrics and "true_label" in metrics:
                pred_dict["probability"] = np.array(metrics["probability"])
                pred_dict["true_label"] = np.array(metrics["true_label"])
            if "val_probability" in metrics and "val_true_label" in metrics:
                pred_dict["val_probability"] = np.array(metrics["val_probability"])
                pred_dict["val_true_label"] = np.array(metrics["val_true_label"])
            if pred_dict:
                fold_seed_predictions[fold_idx].append(pred_dict)

    # --- Standard per-run aggregation ---
    per_run_agg = compute_aggregate_metrics(all_raw_metrics)

    # --- Fold-level seed-ensemble aggregation ---
    fold_ensemble_metrics = _seed_ensemble_metrics(
        fold_seed_predictions, "_fold_idx", ensemble_threshold_fn)
    fold_ensemble_agg = compute_aggregate_metrics(fold_ensemble_metrics)

    return per_run_agg, all_raw_metrics, fold_ensemble_agg


def run_leave_one_subject_out_cv(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, Any]],
    n_seeds_per_fold: int = 1,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Run Leave-One-Subject-Out (LOSO) Cross Validation.

    Each unique subject is used as the test set exactly once.
    The model is trained on all other subjects.
    Metrics are computed on the pooled predictions across all folds.

    Args:
        interviews: List of all interviews.
        train_eval_fn: Callback function `fn(train_pool, test_set, seed)`.
                       For LOSO, this should return a dict containing 'true_label' and 'probability'
                       along with any other metrics.
        n_seeds_per_fold: Number of seeds to run per subject for stability.
        random_state: Base seed for reproducibility.

    Returns:
        A tuple:
        - Aggregated metrics (based on pooled predictions)
        - List of all raw metric dictionaries
    """
    labels = [iv["label"] for iv in interviews]
    groups = [iv["interview_id"] for iv in interviews]
    unique_groups = sorted(list(set(groups)))
    
    cv = LeaveOneGroupOut()

    all_raw_metrics = []
    
    print(f"\n={ '='*70 }=")
    print(f" Starting Leave-One-Subject-Out CV")
    print(f" Total Subjects (Folds): {len(unique_groups)}, Seeds/Fold: {n_seeds_per_fold}")
    print(f" Total training runs: {len(unique_groups) * n_seeds_per_fold}")
    print(f"={ '='*70 }=")

    # Convert to numpy for indexing
    ivs_np = np.array(interviews)

    # To compute pooled metrics per seed
    seed_predictions = {seed_idx: {"y_true": [], "y_prob": []} for seed_idx in range(n_seeds_per_fold)}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(ivs_np, labels, groups), 1):
        train_pool = ivs_np[train_idx].tolist()
        test_set = ivs_np[test_idx].tolist()

        test_sid = unique_groups[fold_idx - 1]
        print(f"\n─── Fold {fold_idx}/{len(unique_groups)} (Subject: {test_sid}) ───")
        print(f"Train subjects: {len(unique_groups)-1}, Records: {len(train_pool)}")
        print(f"Test records: {len(test_set)}")

        for seed_idx in range(n_seeds_per_fold):
            run_seed = random_state + (fold_idx * 100) + seed_idx
            print(f"  [Run {seed_idx+1}/{n_seeds_per_fold}] Seed = {run_seed}")
            
            # The train_eval_fn is expected to return results for the test_set (1 subject)
            results = train_eval_fn(train_pool, test_set, run_seed)
            
            # Support both returning metrics only or full results
            # For LOSO, we NEED predictions to aggregate.
            # Assuming 'evaluate' returns a dict with 'true_label' and 'probability'
            # if they are lists (from multiple records of the same subject), we extend.
            
            if "true_label" in results and "probability" in results:
                y_true = results["true_label"]
                y_prob = results["probability"]
                if not isinstance(y_true, list): y_true = [y_true]
                if not isinstance(y_prob, list): y_prob = [y_prob]
                
                seed_predictions[seed_idx]["y_true"].extend(y_true)
                seed_predictions[seed_idx]["y_prob"].extend(y_prob)

            # Store run metadata
            results["_fold_idx"] = fold_idx
            results["_seed_idx"] = seed_idx
            results["_run_seed"] = run_seed
            all_raw_metrics.append(results)

    # Compute pooled metrics and Bootstrap CIs for each seed
    seed_bootstrap_results = []
    for seed_idx in range(n_seeds_per_fold):
        y_true = np.array(seed_predictions[seed_idx]["y_true"])
        y_prob = np.array(seed_predictions[seed_idx]["y_prob"])
        
        label_dist = f"pos={np.sum(y_true==1)}, neg={np.sum(y_true==0)}"
        print(f"  [Seed {seed_idx}] Computing Bootstrap CIs (N=2000)... ({label_dist})")
        
        boot_metrics = compute_bootstrap_metrics(
            y_true, y_prob, metric_fn=compute_metrics, n_resamples=2000, seed=random_state + seed_idx
        )
        seed_bootstrap_results.append(boot_metrics)

    # Final aggregation: average the bootstrap statistics across seeds
    # This is more robust as it combines subject-level uncertainty with seed stability.
    print("\nAggregating Bootstrap results across seeds...")
    
    metric_keys = list(seed_bootstrap_results[0].keys())
    agg_metrics = {}
    
    for key in metric_keys:
        means = [res[key]["mean"] for res in seed_bootstrap_results]
        lowers = [res[key]["ci_lower"] for res in seed_bootstrap_results]
        uppers = [res[key]["ci_upper"] for res in seed_bootstrap_results]
        stds = [res[key].get("std", 0.0) for res in seed_bootstrap_results]
        
        agg_metrics[key] = {
            "mean": float(np.mean(means)),
            "std": float(np.mean(stds)),
            "ci_lower": float(np.mean(lowers)),
            "ci_upper": float(np.mean(uppers)),
        }
    
    return agg_metrics, all_raw_metrics
