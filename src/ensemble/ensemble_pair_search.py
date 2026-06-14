"""Ensemble roadmap: leakage-free best-PAIR selection over {encoder}x{plain,GRU}.

All members are pos_weight=1.0, 30 seeds, base_seed 100 -> seed-aligned, dev and
test per-seed probabilities saved. Selection uses DEV only (test never touched
during selection). Implements the roadmap:

  1. best-PAIR selection (diversity via encoder x arch), constrained to TWO
     DISTINCT ENCODERS for the final model (fair vs 2-branch fusion baselines).
  2. per-member temperature scaling on DEV before averaging.
  3. logit (log-odds) mean vs arithmetic mean.

Threshold = a-priori PREVALENCE (0.28) everywhere. A dev-tuned F1 threshold was
tried and REJECTED: dev has only 35 subjects, so the F1-optimal dev threshold
degenerates (t~0.15, recall->1) and, when the pair is ALSO selected on dev, the
dev score is doubly optimistic (0.83 dev -> 0.67 test). Prevalence-matching is
the established honest rule here (transductive, no labels) and reproduces the
true test number. Selection metric = DEV macro-F1 at the prevalence threshold.

Pipeline: fit each member's temperature on dev; for every pair x {arith,logit}
score DEV macro-F1 (mean over seeds, prevalence threshold); pick the best config
among DISTINCT-ENCODER pairs; report its TEST metrics (full 30-seed + 6x(5+5))
with one-sample t vs Milintsevich 0.739 and the single bar 0.774. The full dev
ranking (incl. same-encoder pairs, marked) is printed for transparency.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV = 0.28
MILINTSEVICH = 0.739
SINGLE_BAR = 0.774

# name -> (results.json, encoder-id, arch)
MEMBERS = {
    "bge-plain":  ("results/single_model/headline_pw1_30seed/results.json", "bge", "plain"),
    "bge-gru":    ("results/ensemble/pw1_members/bge_gru_pw1/results.json", "bge", "gru"),
    "mxbai-plain":("results/ensemble/pw1_members/mxbai_plain_pw1/results.json", "mxbai", "plain"),
    "mxbai-gru":  ("results/ensemble/pw1_members/official_mxbai_gru_pw1_30seed/results.json", "mxbai", "gru"),
    "uae-plain":  ("results/ensemble/pw1_members/uae_plain_pw1/results.json", "uae", "plain"),
    "uae-gru":    ("results/ensemble/pw1_members/uae_gru_pw1/results.json", "uae", "gru"),
}

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(dev_runs, y):
    """Scalar T>0 minimising mean dev NLL over seeds (probs -> logits/T)."""
    z = [_logit(np.asarray(p)) for p in dev_runs]
    y = np.asarray(y, dtype=float)

    def nll(T):
        T = max(T, 1e-3)
        tot = 0.0
        for zi in z:
            p = np.clip(_sig(zi / T), EPS, 1 - EPS)
            tot += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return tot / len(z)

    r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def load_member(path):
    r = json.load(open(path))
    return {
        "dev": [np.asarray(p) for p in r["dev_prob_runs"]],
        "test": [np.asarray(p) for p in r["test_prob_runs"]],
        "ydev": np.asarray(r["dev_labels"]),
        "ytest": np.asarray(r["test_labels"]),
    }


def combine(pa, pb, rule):
    if rule == "logit":
        return _sig((_logit(pa) + _logit(pb)) / 2)
    return (pa + pb) / 2


def per_seed_metrics(runs_a, runs_b, y, rule, thr_mode):
    """Mean metrics over seed-aligned pairs at the chosen combine rule/threshold."""
    out = []
    n = min(len(runs_a), len(runs_b))
    for i in range(n):
        prob = combine(runs_a[i], runs_b[i], rule)
        if thr_mode == "prevalence":
            t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        else:  # dev-tuned f1 done by caller on dev probs; here reuse prevalence on test
            t = thr_mode  # numeric threshold passed in
        out.append(compute_metrics(y, (prob >= t).astype(int), prob))
    return out


def dev_score(m_a, m_b, T_a, T_b, rule):
    """Mean dev macro-F1 over seeds at the prevalence threshold."""
    a = [_sig(_logit(p) / T_a) for p in m_a["dev"]]
    b = [_sig(_logit(p) / T_b) for p in m_b["dev"]]
    y = m_a["ydev"]
    vals = []
    for i in range(min(len(a), len(b))):
        prob = combine(a[i], b[i], rule)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        vals.append(compute_metrics(y, (prob >= t).astype(int), prob)["macro_f1"])
    return float(np.mean(vals))


def report_test(m_a, m_b, T_a, T_b, rule):
    a = [_sig(_logit(p) / T_a) for p in m_a["test"]]
    b = [_sig(_logit(p) / T_b) for p in m_b["test"]]
    y = m_a["ytest"]
    n = min(len(a), len(b))

    # full 30-seed
    em = []
    for i in range(n):
        prob = combine(a[i], b[i], rule)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        em.append(compute_metrics(y, (prob >= t).astype(int), prob))
    print(f"  full {n}-seed ensemble:")
    for k in ("macro_f1", "micro_f1", "roc_auc"):
        v = np.array([m[k] for m in em])
        print(f"    {k}: {v.mean():.3f} ± {v.std():.3f}")
    macro = np.array([m["macro_f1"] for m in em])
    print(f"    t vs Milintsevich {MILINTSEVICH}: p={stats.ttest_1samp(macro, MILINTSEVICH)[1]:.3g}")
    p2 = stats.ttest_1samp(macro, SINGLE_BAR)[1]
    print(f"    t vs single bar {SINGLE_BAR}: Δ={macro.mean()-SINGLE_BAR:+.3f} p={p2:.3g} "
          f"({'clears' if macro.mean() > SINGLE_BAR and p2 < 0.05 else 'no robust gain'})")

    # 6x(5+5)
    g = n // 5
    grp = []
    for gi in range(g):
        idx = range(gi * 5, gi * 5 + 5)
        prob = np.mean([combine(a[i], b[i], rule) for i in idx], axis=0)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        grp.append(compute_metrics(y, (prob >= t).astype(int), prob))
    print(f"  {g}x(5+5) independent:")
    for k in ("macro_f1", "micro_f1", "roc_auc"):
        v = np.array([m[k] for m in grp])
        print(f"    {k}: {v.mean():.3f} ± {v.std():.3f}")
    gm = np.array([m["macro_f1"] for m in grp])
    print(f"    t vs Milintsevich {MILINTSEVICH}: p={stats.ttest_1samp(gm, MILINTSEVICH)[1]:.3g}")


def main():
    avail = {}
    for name, (path, enc, arch) in MEMBERS.items():
        if Path(path).exists():
            m = load_member(path); m["enc"] = enc; m["arch"] = arch
            m["T"] = fit_temperature(m["dev"], m["ydev"])
            avail[name] = m
        else:
            print(f"  [missing] {name}: {path}")
    print(f"\nmembers available: {list(avail.keys())}")
    for n, m in avail.items():
        print(f"  {n:12s} T={m['T']:.3f} enc={m['enc']} arch={m['arch']}")
    # consistent test labels
    names = list(avail)
    y0 = avail[names[0]]["ytest"]
    for n in names[1:]:
        assert np.array_equal(y0, avail[n]["ytest"]), f"label mismatch {n}"

    print("\nDEV macro-F1 (prevalence threshold) over all pairs x rule "
          "(* = distinct-encoder, eligible for final):")
    results = []
    for a, b in combinations(names, 2):
        ma, mb = avail[a], avail[b]
        distinct = ma["enc"] != mb["enc"]
        for rule in ("arith", "logit"):
            s = dev_score(ma, mb, ma["T"], mb["T"], rule)
            results.append((s, a, b, rule, distinct))
    results.sort(reverse=True)
    for s, a, b, rule, distinct in results:
        mark = "*" if distinct else " "
        print(f"  {mark} {s:.3f}  {a:12s}+{b:12s} {rule:5s}")

    # best DISTINCT-encoder config = final (2-encoder constraint)
    best = next(r for r in results if r[4])
    s, a, b, rule, _ = best
    print(f"\n=== SELECTED (dev macro-F1 {s:.3f}, distinct encoders) ===")
    print(f"  pair = {a} + {b} | rule = {rule} | threshold = prevalence")
    print("  TEST metrics:")
    report_test(avail[a], avail[b], avail[a]["T"], avail[b]["T"], rule)


if __name__ == "__main__":
    main()
