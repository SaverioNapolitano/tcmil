"""Statistical utilities for metric aggregation and confidence intervals."""

import numpy as np
import scipy.stats as stats


def compute_aggregate_metrics(
    metrics_list: list[dict[str, float]],
    confidence_level: float = 0.95
) -> dict[str, dict[str, float]]:
    """Compute mean, standard deviation, and confidence intervals across multiple runs.

    Args:
        metrics_list: List of metric dictionaries from independent runs.
        confidence_level: Confidence level for the interval (e.g., 0.95 for 95%).

    Returns:
        A dictionary mapping each metric name to a dictionary of statistics:
            {
                "f1": {"mean": 0.65, "std": 0.02, "ci_lower": 0.61, "ci_upper": 0.69},
                ...
            }
    """
    if not metrics_list:
        return {}

    # Extract all keys from the first dictionary, assuming consistent keys
    # Ignore non-numeric metrics like 'confusion_matrix' if present
    metric_keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    
    n = len(metrics_list)
    results = {}

    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
            
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if n > 1 else 0.0

        if n > 1 and std_val > 0:
            # Using t-distribution for small sample sizes
            se = std_val / np.sqrt(n)
            h = se * stats.t.ppf((1 + confidence_level) / 2., n - 1)
            ci_lower = mean_val - h
            ci_upper = mean_val + h
        else:
            ci_lower = mean_val
            ci_upper = mean_val

        results[key] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    return results


def format_aggregate_report(agg_metrics: dict[str, dict[str, float]]) -> str:
    """Format aggregated metrics into a human-readable string."""
    lines = []
    lines.append(f"{'Metric':<20} | {'Mean':<8} | {'Std':<8} | {'95% CI'}")
    lines.append("-" * 65)
    
    for metric, stats_dict in agg_metrics.items():
        mean = stats_dict["mean"]
        std = stats_dict["std"]
        ci_l = stats_dict["ci_lower"]
        ci_u = stats_dict["ci_upper"]
        
        lines.append(
            f"{metric:<20} | {mean:<8.4f} | {std:<8.4f} | [{ci_l:.4f}, {ci_u:.4f}]"
        )
        
    return "\n".join(lines)


def compare_methods_cv(
    runs_a: list[dict[str, float]],
    runs_b: list[dict[str, float]],
    confidence_level: float = 0.95
) -> dict[str, dict[str, float]]:
    """Compare two methods based on their raw CV runs using paired split-wise differences.

    1. Groups runs by `_split_idx` and averages across seeds for each split.
    2. Computes paired split-wise differences (A - B).
    3. Calculates Wilcoxon signed-rank test p-value, 95% CI of the difference, and win rate.

    Args:
        runs_a: Raw runs from method A (from run_monte_carlo_cv).
        runs_b: Raw runs from method B (from run_monte_carlo_cv).
        confidence_level: Confidence level for the difference interval.

    Returns:
        Dictionary of comparison metrics for each key.
    """
    if not runs_a or not runs_b:
        return {}

    def _aggregate_by_split(runs):
        splits = {}
        for r in runs:
            s_idx = r["_split_idx"]
            if s_idx not in splits:
                splits[s_idx] = []
            splits[s_idx].append(r)
        
        split_means = {}
        for s_idx, split_runs in splits.items():
            metric_keys = [k for k in split_runs[0].keys() if isinstance(split_runs[0][k], (int, float))]
            split_means[s_idx] = {}
            for k in metric_keys:
                if k.startswith("_"):
                    continue
                split_means[s_idx][k] = np.mean([r[k] for r in split_runs if k in r])
        return split_means

    means_a = _aggregate_by_split(runs_a)
    means_b = _aggregate_by_split(runs_b)

    common_splits = sorted(list(set(means_a.keys()) & set(means_b.keys())))
    if not common_splits:
        return {}

    metric_keys = [k for k in means_a[common_splits[0]].keys()]
    n = len(common_splits)
    
    results = {}
    for key in metric_keys:
        vals_a = np.array([means_a[s][key] for s in common_splits])
        vals_b = np.array([means_b[s][key] for s in common_splits])
        
        diff = vals_a - vals_b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if n > 1 else 0.0

        if n > 1 and std_diff > 0:
            se = std_diff / np.sqrt(n)
            h = se * stats.t.ppf((1 + confidence_level) / 2., n - 1)
            ci_lower = mean_diff - h
            ci_upper = mean_diff + h
        else:
            ci_lower = mean_diff
            ci_upper = mean_diff

        # Wilcoxon signed-rank test
        try:
            # alternative='two-sided' by default
            # Add zero_method='zsplit' to handle zero differences gracefully
            # Wilcoxon requires n>=10 for valid p-values under normal approx, but scipy handles small n exactly.
            if np.all(diff == 0):
                p_value = 1.0
            else:
                _, p_value = stats.wilcoxon(vals_a, vals_b)
        except ValueError:
            p_value = float('nan')

        win_rate = np.sum(diff > 0) / n
        tie_rate = np.sum(diff == 0) / n
        loss_rate = np.sum(diff < 0) / n

        results[key] = {
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "win_rate": float(win_rate),
            "tie_rate": float(tie_rate),
            "loss_rate": float(loss_rate),
            "n_splits": n,
        }

    return results


def format_comparison_report(comp_metrics: dict[str, dict[str, float]], method_a_name: str = "A", method_b_name: str = "B") -> str:
    """Format comparison metrics into a human-readable string."""
    lines = []
    lines.append(f"Comparison: {method_a_name} vs {method_b_name}")
    lines.append(f"Positive differences mean {method_a_name} is better. Win Rate = % of splits where {method_a_name} > {method_b_name}.")
    lines.append("")
    lines.append(f"{'Metric':<20} | {'Mean Diff':<10} | {'95% CI of Diff':<18} | {'p-value':<8} | {'Win Rate':<8}")
    lines.append("-" * 75)
    
    for metric, stats_dict in comp_metrics.items():
        mdiff = stats_dict["mean_diff"]
        ci_l = stats_dict["ci_lower"]
        ci_u = stats_dict["ci_upper"]
        pval = stats_dict["p_value"]
        win = stats_dict["win_rate"]
        
        pval_str = f"{pval:.4f}" if not np.isnan(pval) else "NaN"
        
        lines.append(
            f"{metric:<20} | {mdiff:>10.4f} | [{ci_l:>6.4f}, {ci_u:>6.4f}] | {pval_str:>8} | {win:>8.2%}"
        )
        
    return "\n".join(lines)


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute confidence intervals for metrics using bootstrap resampling.

    This provides an estimate of generalization uncertainty across the 
    sample population (subjects), which is more meaningful in LOSO CV 
    than seed-based variability.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        metric_fn: Function that takes (y_true, y_pred, y_prob) and returns a dict of metrics.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g., 0.95 for 95%).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping each metric name to its bootstrap statistics:
            {"f1": {"mean": 0.5, "std": 0.05, "ci_lower": 0.45, "ci_upper": 0.55}, ...}
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    
    # Store results for each bootstrap sample
    bootstrap_results = []
    
    # The 'mean' performance is the one on the full original sample
    y_pred_orig = (y_prob >= 0.5).astype(int)
    original_metrics = metric_fn(y_true, y_pred_orig, y_prob)
    
    for _ in range(n_resamples):
        # Resample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        y_true_resampled = y_true[indices]
        y_prob_resampled = y_prob[indices]
        y_pred_resampled = (y_prob_resampled >= 0.5).astype(int)
        
        # Calculate metrics for this resample
        resampled_metrics = metric_fn(y_true_resampled, y_pred_resampled, y_prob_resampled)
        bootstrap_results.append(resampled_metrics)
        
    final_results = {}
    metric_keys = original_metrics.keys()
    
    alpha = 1.0 - confidence_level
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100
    
    for key in metric_keys:
        if not isinstance(original_metrics[key], (int, float)) or np.isnan(original_metrics[key]):
            continue
            
        # Filter out NaNs (e.g., ROC-AUC if a resample has only one class)
        resampled_values = [r[key] for r in bootstrap_results if not np.isnan(r.get(key, np.nan))]
        if not resampled_values:
            continue
            
        ci_lower = np.percentile(resampled_values, lower_p)
        ci_upper = np.percentile(resampled_values, upper_p)
        
        final_results[key] = {
            "mean": original_metrics[key],
            "std": float(np.std(resampled_values, ddof=1)) if len(resampled_values) > 1 else 0.0,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
        
    return final_results

