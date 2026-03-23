"""Generic Monte Carlo Cross Validation framework.

Supports repeated stratified splits at the subject/interview level,
with multiple random seeds per split, for stable and trustworthy evaluation.
"""

from typing import Callable, Any
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils.stats import compute_aggregate_metrics


def run_monte_carlo_cv(
    interviews: list[dict[str, Any]],
    train_eval_fn: Callable[[list[dict], list[dict], int], dict[str, float]],
    n_splits: int = 5,
    n_seeds_per_split: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Run Repeated Stratified Monte Carlo Cross Validation.

    For each split, the data is partitioned into a training pool and a hold-out test set.
    The true test set is strictly preserved and never leaked.

    The user-provided `train_eval_fn` is responsible for:
      1. Internally carving out a validaton set from the training pool (if doing early stopping).
      2. Training the model using the provided random seed.
      3. Returning a dictionary of metrics computed STRICTLY on the test set.

    Args:
        interviews: List of all interviews (train + dev + test combined).
        train_eval_fn: Callback function `fn(train_pool, test_set, seed) -> metrics_dict`.
        n_splits: Number of random monte carlo splits to generate.
        n_seeds_per_split: Number of independent training runs per split.
        test_size: Proportion of data to hold out for the test set.
        random_state: Seed for the StratifiedShuffleSplit generator.

    Returns:
        A tuple:
        - Aggregated metrics (mean, std, 95% CI)
        - List of all raw metric dictionaries from every run
    """
    labels = [iv["label"] for iv in interviews]
    
    cv = StratifiedShuffleSplit(
        n_splits=n_splits, 
        test_size=test_size, 
        random_state=random_state
    )

    all_raw_metrics = []
    
    print(f"\n={ '='*70 }=")
    print(f" Starting Monte Carlo CV")
    print(f" Splits: {n_splits}, Seeds per split: {n_seeds_per_split}, Test size: {test_size}")
    print(f" Total training runs: {n_splits * n_seeds_per_split}")
    print(f"={ '='*70 }=")

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        train_pool = [interviews[i] for i in train_idx]
        test_set = [interviews[i] for i in test_idx]
        
        print(f"\n─── Split {split_idx + 1}/{n_splits} ───")
        print(f"Train pool size: {len(train_pool)}")
        print(f"Test set size: {len(test_set)}")

        for seed_idx in range(n_seeds_per_split):
            # Deterministic but varied seed for each run
            run_seed = random_state + (split_idx * 100) + seed_idx
            
            print(f"\n  [Split {split_idx + 1}, Run {seed_idx + 1}/{n_seeds_per_split}] Seed = {run_seed}")
            
            # The callback must do all the heavy lifting:
            # feature extraction, inner-validation splitting, training, and testing.
            metrics = train_eval_fn(train_pool, test_set, run_seed)
            
            # Store metadata with the run
            metrics["_split_idx"] = split_idx
            metrics["_seed_idx"] = seed_idx
            metrics["_run_seed"] = run_seed
            
            all_raw_metrics.append(metrics)
            
            # Print a quick summary of this run's primary metrics
            f1 = metrics.get('f1', 0.0)
            bacc = metrics.get('balanced_accuracy', 0.0)
            print(f"  -> Result: F1={f1:.4f}, B.Acc={bacc:.4f}")

    print("\nAggregating results across all runs...")
    agg_metrics = compute_aggregate_metrics(all_raw_metrics)
    
    return agg_metrics, all_raw_metrics