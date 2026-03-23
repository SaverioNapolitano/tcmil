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
