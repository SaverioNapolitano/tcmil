"""B4 eval: heterogeneous bag-view ensemble bge-gru@w4 + uae-gru@w6.

Equal-weight, temperature-scaled, arithmetic mean, prevalence threshold (the
robust recipe; A showed re-weighting overfits). Compares against the current
best bge-gru@w4 + uae-gru@w4 (test 6x5+5 macro 0.833) and reports the
between-member prob correlation for each pairing — B4 only helps if the w6 view
decorrelates the members.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV, BAR = 0.28, 0.774
EPS = 1e-6
P = {
    "bge-gru-w4": "results/ensemble/pw1_members/bge_gru_pw1/results.json",
    "uae-gru-w4": "results/ensemble/pw1_members/uae_gru_pw1/results.json",
    "uae-gru-w6": "results/ensemble/pw1_members/uae_gru_w6_pw1/results.json",
    "uae-gru-ponly": "results/ensemble/pw1_members/uae_gru_ponly_pw1/results.json",
}


def lg(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sg(z): return 1 / (1 + np.exp(-z))


def load(p):
    r = json.load(open(p))
    return ([np.asarray(x) for x in r["dev_prob_runs"]],
            [np.asarray(x) for x in r["test_prob_runs"]],
            np.asarray(r["dev_labels"]), np.asarray(r["test_labels"]))


def temp(dev, y):
    from scipy.optimize import minimize_scalar
    z = [lg(p) for p in dev]; y = y.astype(float)
    def f(T):
        T = max(T, 1e-3); t = 0.0
        for zi in z:
            p = np.clip(sg(zi / T), EPS, 1 - EPS)
            t += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return t / len(z)
    return minimize_scalar(f, bounds=(0.05, 20), method="bounded").x


def evalpair(a_runs, b_runs, Ta, Tb, y, tag, ref):
    A = [sg(lg(p) / Ta) for p in a_runs]; B = [sg(lg(p) / Tb) for p in b_runs]
    n = min(len(A), len(B))
    # between-member correlation on test (mean over seeds of per-subject corr)
    corr = np.mean([np.corrcoef(A[i], B[i])[0, 1] for i in range(n)])
    em = []
    for i in range(n):
        prob = 0.5 * A[i] + 0.5 * B[i]
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        em.append(compute_metrics(y, (prob >= t).astype(int), prob))
    g = n // 5; grp = []
    for gi in range(g):
        prob = np.mean([0.5 * A[i] + 0.5 * B[i] for i in range(gi*5, gi*5+5)], axis=0)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        grp.append(compute_metrics(y, (prob >= t).astype(int), prob))
    f30 = np.array([m["macro_f1"] for m in em])
    g6 = {k: np.array([m[k] for m in grp]) for k in ("macro_f1","micro_f1","roc_auc")}
    print(f"\n=== {tag} ===  member prob-corr={corr:.3f}")
    print(f"  full-30 macro {f30.mean():.3f}±{f30.std():.3f}")
    print(f"  6x(5+5) macro {g6['macro_f1'].mean():.3f}±{g6['macro_f1'].std():.3f} "
          f"micro {g6['micro_f1'].mean():.3f} auc {g6['roc_auc'].mean():.3f}")
    print(f"  vs baseline {ref}: Δ6x={g6['macro_f1'].mean()-ref:+.3f}")
    return g6["macro_f1"].mean()


def main():
    bge = load(P["bge-gru-w4"]); uae4 = load(P["uae-gru-w4"]); uae6 = load(P["uae-gru-w6"])
    Tb = temp(bge[0], bge[2]); Tu4 = temp(uae4[0], uae4[2]); Tu6 = temp(uae6[0], uae6[2])
    y = bge[3]
    base = evalpair(bge[1], uae4[1], Tb, Tu4, y, "baseline bge-gru@w4 + uae-gru@w4", 0.833)
    evalpair(bge[1], uae6[1], Tb, Tu6, y, "B4 bge-gru@w4 + uae-gru@w6", base)
    if Path(P["uae-gru-ponly"]).exists():
        upo = load(P["uae-gru-ponly"]); Tpo = temp(upo[0], upo[2])
        evalpair(bge[1], upo[1], Tb, Tpo, y, "B5 bge-gru@w4(dialogue) + uae-gru(ponly)", base)


if __name__ == "__main__":
    main()
