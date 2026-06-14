"""Statistical tests for the TC-MIL results.

1. Bootstrap 95% CI on the official-test ensemble (subject resampling at the
   final decision threshold — uncertainty over the 46 test subjects).
2. Wilcoxon signed-rank test between two CV runs on matched (fold, seed)
   per-run metrics — paired, because both runs share fold assignments and
   seeds (same --seed and fold construction).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.utils.metrics import compute_metrics


def bootstrap_official(member_dirs, threshold, n_resamples=2000, seed=42):
    test_probs = []
    for d in member_dirs:
        r = json.load(open(Path(d) / "results.json"))
        test_probs.append(np.array(r["test_probs"]))
        y_true = np.array(r["test_labels"])
    y_prob = np.mean(test_probs, axis=0)

    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
    boots = []
    for _ in range(n_resamples):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(compute_metrics(
            y_true[idx], (y_prob[idx] >= threshold).astype(int), y_prob[idx]))

    print(f"\nOfficial test bootstrap ({len(member_dirs)} members, t={threshold}, "
          f"N={len(y_true)}, {len(boots)} resamples):")
    for k in ["roc_auc", "f1", "precision", "recall", "balanced_accuracy"]:
        vals = np.array([b[k] for b in boots])
        print(f"  {k:<18} {point[k]:.4f}  [{np.percentile(vals, 2.5):.4f}, "
              f"{np.percentile(vals, 97.5):.4f}]")


def wilcoxon_cv(run_a, run_b, label_a, label_b):
    def per_run(d):
        r = json.load(open(Path(d) / "cv_results.json"))
        out = {}
        for m in r["raw"]:
            key = (m.get("_repeat", 0), m["_fold_idx"], m["_seed_idx"])
            out[key] = m
        return out

    a, b = per_run(run_a), per_run(run_b)
    keys = sorted(set(a) & set(b))
    if len(keys) < len(a) or len(keys) < len(b):
        print(f"  [warn] only {len(keys)} matched runs "
              f"({len(a)} vs {len(b)} available)")

    print(f"\nWilcoxon signed-rank, {label_a} vs {label_b} "
          f"({len(keys)} paired runs):")
    for metric in ["roc_auc", "f1", "balanced_accuracy"]:
        xa = np.array([a[k][metric] for k in keys])
        xb = np.array([b[k][metric] for k in keys])
        stat, p = wilcoxon(xa, xb)
        print(f"  {metric:<18} {label_a}={xa.mean():.4f} {label_b}={xb.mean():.4f} "
              f"diff={xa.mean()-xb.mean():+.4f}  W={stat:.1f} p={p:.4g}")


def main():
    p = argparse.ArgumentParser(description="TC-MIL statistical tests")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="bootstrap CI on official test ensemble")
    b.add_argument("--members", nargs="+", required=True,
                   help="oof_threshold_official.py output dirs (probs averaged)")
    b.add_argument("--threshold", type=float, required=True)

    w = sub.add_parser("wilcoxon", help="paired test between two CV runs")
    w.add_argument("--run_a", required=True)
    w.add_argument("--run_b", required=True)
    w.add_argument("--label_a", default="A")
    w.add_argument("--label_b", default="B")

    args = p.parse_args()
    if args.cmd == "bootstrap":
        bootstrap_official(args.members, args.threshold)
    else:
        wilcoxon_cv(args.run_a, args.run_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
