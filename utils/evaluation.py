"""Generic Monte Carlo Cross Validation framework.

Supports repeated stratified splits at the subject/interview level,
with multiple random seeds per split, for stable and trustworthy evaluation.
"""

from typing import Callable, Any
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold

from utils.stats import compute_aggregate_metrics


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
