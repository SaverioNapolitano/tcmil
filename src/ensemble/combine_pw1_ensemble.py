"""Offline pos_weight=1.0 ensemble: bge-large plain + mxbai GRU.

Both members are pos_weight=1.0, 30 seeds, base_seed 100 -> seed-aligned, so
averaging the saved per-seed test probabilities needs no retraining. Mirrors the
parked auto-member 6x(5+5) protocol, plus the full 30-seed average, evaluated at
the a-priori prevalence (testprev) threshold to match the single-model headline.

Compares against the NEW single-model bar (macro-F1 0.774), not the old 0.751.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV = 0.28
MILINTSEVICH = 0.739
SINGLE_PW1_BAR = 0.774  # new single-model macro-F1

MEMBER_A = "results/single_model/headline_pw1_30seed/results.json"          # bge plain pw1.0
MEMBER_B = "results/ensemble/pw1_members/official_mxbai_gru_pw1_30seed/results.json"  # mxbai GRU pw1.0


def per_seed_metrics(prob_runs, y):
    out = []
    for prob in prob_runs:
        prob = np.asarray(prob)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        out.append(compute_metrics(y, (prob >= t).astype(int), prob))
    return out


def main():
    ra, rb = json.load(open(MEMBER_A)), json.load(open(MEMBER_B))
    y = np.asarray(ra["test_labels"])
    assert np.array_equal(y, np.asarray(rb["test_labels"])), "label order mismatch"
    A = [np.asarray(p) for p in ra["test_prob_runs"]]
    B = [np.asarray(p) for p in rb["test_prob_runs"]]
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    print(f"members: bge-plain pw1.0 (n={len(A)}) + mxbai-GRU pw1.0 (n={len(B)}); "
          f"seed-aligned base_seed 100\n")

    # --- Full 30-seed ensemble: average matched-seed prob pairs ---
    ens = [(A[i] + B[i]) / 2 for i in range(n)]
    em = per_seed_metrics(ens, y)
    for k in ("precision", "recall", "f1", "macro_f1", "micro_f1", "roc_auc"):
        v = np.array([m[k] for m in em])
        print(f"  full {n}-seed ensemble {k}: {v.mean():.3f} ± {v.std():.3f}")
    macro = np.array([m["macro_f1"] for m in em])
    t, p = stats.ttest_1samp(macro, MILINTSEVICH)
    print(f"  one-sample t vs Milintsevich {MILINTSEVICH}: p={p:.3g}")
    t2, p2 = stats.ttest_1samp(macro, SINGLE_PW1_BAR)
    print(f"  one-sample t vs single-pw1 bar {SINGLE_PW1_BAR}: "
          f"Δ={macro.mean()-SINGLE_PW1_BAR:+.3f} p={p2:.3g} "
          f"({'clears' if macro.mean() > SINGLE_PW1_BAR and p2 < 0.05 else 'no robust gain'})")

    # --- 6x(5+5): 6 disjoint groups of 5 seeds; within a group average the
    #     5 bge + 5 mxbai member probs (10 vectors) -> 1 ensemble -> metrics ---
    g = n // 5
    print(f"\n  {g}x(5+5) independent ensembles:")
    grp = []
    for gi in range(g):
        idx = range(gi * 5, gi * 5 + 5)
        stack = [A[i] for i in idx] + [B[i] for i in idx]
        prob = np.mean(stack, axis=0)
        tt = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        grp.append(compute_metrics(y, (prob >= tt).astype(int), prob))
    for k in ("precision", "recall", "f1", "macro_f1", "micro_f1", "roc_auc"):
        v = np.array([m[k] for m in grp])
        print(f"    {k}: {v.mean():.3f} ± {v.std():.3f}")
    gm = np.array([m["macro_f1"] for m in grp])
    _, pg = stats.ttest_1samp(gm, MILINTSEVICH)
    print(f"    one-sample t vs Milintsevich {MILINTSEVICH}: p={pg:.3g}")


if __name__ == "__main__":
    main()
