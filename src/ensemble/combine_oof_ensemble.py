"""Combine per-encoder OOF runs into a multi-encoder ensemble.

Each input directory is the output of oof_threshold_official.py for one
encoder (same pool, same OOF fold seed, so the OOF probabilities are aligned
by sorted interview id and the test probabilities by split order). The
ensemble averages probabilities across encoders at both stages:

    1. average OOF probs  -> tune the decision threshold (per strategy)
    2. average test probs -> evaluate once at those thresholds

This keeps the OOF property: every threshold statistic comes from models
that never trained on the subject being scored, and test labels are never
used for tuning.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics


def load_run(d: Path):
    r = json.load(open(d / "results.json"))
    return {
        "name": d.name,
        "oof_probs": np.array(r["oof_probs"]),
        "oof_labels": np.array(r["oof_labels"]),
        "test_probs": np.array(r["test_probs"]),
        "test_labels": np.array(r["test_labels"]),
        "pool_prev": r["pool_prevalence"],
    }


def evaluate(runs, label):
    oof_labels = runs[0]["oof_labels"]
    test_labels = runs[0]["test_labels"]
    for r in runs[1:]:
        assert np.array_equal(r["oof_labels"], oof_labels), "OOF label misalignment"
        assert np.array_equal(r["test_labels"], test_labels), "test label misalignment"

    oof_avg = np.mean([r["oof_probs"] for r in runs], axis=0)
    test_avg = np.mean([r["test_probs"] for r in runs], axis=0)
    pool_prev = runs[0]["pool_prev"]

    oof_auc = compute_metrics(oof_labels, (oof_avg >= 0.5).astype(int), oof_avg)["roc_auc"]
    out = {"members": [r["name"] for r in runs], "oof_auc": float(oof_auc), "test_by_strategy": {}}
    print(f"\n--- {label} (OOF AUC {oof_auc:.4f}) ---")
    for metric in ["f1", "bacc", "youden", "prevalence"]:
        t = tune_threshold(oof_labels, oof_avg, metric, prevalence=pool_prev)
        tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        out["test_by_strategy"][metric] = {"threshold": float(t), **tm}
        print(f"  [OOF-{metric:<10}] t={t:.2f}  F1={tm['f1']:.4f} P={tm['precision']:.4f} "
              f"R={tm['recall']:.4f} BAcc={tm['balanced_accuracy']:.4f} AUC={tm['roc_auc']:.4f}")
    return out


def main():
    p = argparse.ArgumentParser(description="Multi-encoder OOF-threshold ensemble")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Output dirs of oof_threshold_official.py runs to combine.")
    p.add_argument("--output", default="results/tcmil_oof_ensemble/results.json")
    args = p.parse_args()

    runs = [load_run(Path(d)) for d in args.runs]
    results = {"singles": {}, "combos": {}}

    for r in runs:
        results["singles"][r["name"]] = evaluate([r], r["name"])
    for k in range(2, len(runs) + 1):
        for combo in combinations(runs, k):
            label = "+".join(c["name"] for c in combo)
            results["combos"][label] = evaluate(list(combo), label)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
