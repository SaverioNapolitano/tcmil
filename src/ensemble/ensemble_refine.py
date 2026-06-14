"""Ensemble roadmap A (offline): weighted-alpha + Platt vs temperature.

Extends the best-pair search with two combination refinements that need only the
saved dev/test probabilities (no retraining, no OOF):
  A1 weighted alpha  : member_a weight on a dev grid instead of fixed 0.5.
  A3 Platt scaling   : 2-param (scale+bias) per-member calibration vs the
                       1-param temperature used before.
Search = distinct-encoder pairs x calib{temp,platt} x rule{arith,logit} x
alpha-grid, selected on DEV macro-F1 (prevalence threshold), reported on TEST.
The prior best (bge-gru+uae-gru, temp, arith, alpha=0.5 -> test 6x5+5 macro
0.833) is printed as the baseline to beat.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV, MIL, BAR = 0.28, 0.739, 0.774
EPS = 1e-6
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
MEMBERS = {
    "bge-plain":  ("results/single_model/headline_pw1_30seed/results.json", "bge"),
    "bge-gru":    ("results/ensemble/pw1_members/bge_gru_pw1/results.json", "bge"),
    "mxbai-plain":("results/ensemble/pw1_members/mxbai_plain_pw1/results.json", "mxbai"),
    "mxbai-gru":  ("results/ensemble/pw1_members/official_mxbai_gru_pw1_30seed/results.json", "mxbai"),
    "uae-plain":  ("results/ensemble/pw1_members/uae_plain_pw1/results.json", "uae"),
    "uae-gru":    ("results/ensemble/pw1_members/uae_gru_pw1/results.json", "uae"),
}


def _logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def _sig(z): return 1 / (1 + np.exp(-z))


def fit_temp(dev, y):
    z = [_logit(np.asarray(p)) for p in dev]; y = np.asarray(y, float)
    def nll(T):
        T = max(T, 1e-3); t = 0.0
        for zi in z:
            p = np.clip(_sig(zi / T), EPS, 1 - EPS)
            t += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return t / len(z)
    return ("temp", float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x))


def fit_platt(dev, y):
    z = [_logit(np.asarray(p)) for p in dev]; y = np.asarray(y, float)
    def nll(ab):
        a, b = ab; t = 0.0
        for zi in z:
            p = np.clip(_sig(a * zi + b), EPS, 1 - EPS)
            t += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return t / len(z)
    r = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
    return ("platt", (float(r.x[0]), float(r.x[1])))


def cal(p, c):
    z = _logit(np.asarray(p))
    return _sig(z / c[1]) if c[0] == "temp" else _sig(c[1][0] * z + c[1][1])


def combine(pa, pb, rule, a):
    if rule == "logit":
        return _sig(a * _logit(pa) + (1 - a) * _logit(pb))
    return a * pa + (1 - a) * pb


def macro_runs(A, B, y, rule, a):
    out = []
    for i in range(min(len(A), len(B))):
        prob = combine(A[i], B[i], rule, a)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        out.append(compute_metrics(y, (prob >= t).astype(int), prob)["macro_f1"])
    return float(np.mean(out))


def report(A, B, y, rule, a, tag):
    n = min(len(A), len(B))
    em = []
    for i in range(n):
        prob = combine(A[i], B[i], rule, a)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        em.append(compute_metrics(y, (prob >= t).astype(int), prob))
    g = n // 5; grp = []
    for gi in range(g):
        prob = np.mean([combine(A[i], B[i], rule, a) for i in range(gi*5, gi*5+5)], axis=0)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        grp.append(compute_metrics(y, (prob >= t).astype(int), prob))
    print(f"\n=== {tag} ===")
    for lab, ms in [("full-30", em), ("6x(5+5)", grp)]:
        s = {k: np.array([m[k] for m in ms]) for k in ("macro_f1","micro_f1","roc_auc")}
        print(f"  {lab}: macro {s['macro_f1'].mean():.3f}±{s['macro_f1'].std():.3f} "
              f"micro {s['micro_f1'].mean():.3f} auc {s['roc_auc'].mean():.3f}")
    mac = np.array([m["macro_f1"] for m in em])
    p2 = stats.ttest_1samp(mac, BAR)[1]
    print(f"  full-30 vs single {BAR}: Δ={mac.mean()-BAR:+.3f} p={p2:.3g}")


def main():
    M = {}
    for name, (path, enc) in MEMBERS.items():
        if not Path(path).exists():
            print(f"[missing] {name}"); continue
        r = json.load(open(path))
        M[name] = {"enc": enc,
                   "dev": [np.asarray(p) for p in r["dev_prob_runs"]],
                   "test": [np.asarray(p) for p in r["test_prob_runs"]],
                   "yd": np.asarray(r["dev_labels"]), "yt": np.asarray(r["test_labels"])}
        M[name]["temp"] = fit_temp(M[name]["dev"], M[name]["yd"])
        M[name]["platt"] = fit_platt(M[name]["dev"], M[name]["yd"])
    names = list(M)

    # baseline: bge-gru+uae-gru, temp, arith, alpha=0.5
    a0 = [cal(p, M["bge-gru"]["temp"]) for p in M["bge-gru"]["test"]]
    b0 = [cal(p, M["uae-gru"]["temp"]) for p in M["uae-gru"]["test"]]
    report(a0, b0, M["bge-gru"]["yt"], "arith", 0.5, "BASELINE bge-gru+uae-gru temp arith a=0.5")

    best = None
    for x, yname in combinations(names, 2):
        if M[x]["enc"] == M[yname]["enc"]:
            continue
        for calib in ("temp", "platt"):
            Ad = [cal(p, M[x][calib]) for p in M[x]["dev"]]
            Bd = [cal(p, M[yname][calib]) for p in M[yname]["dev"]]
            yd = M[x]["yd"]
            for rule in ("arith", "logit"):
                for a in ALPHAS:
                    s = macro_runs(Ad, Bd, yd, rule, a)
                    if best is None or s > best[0]:
                        best = (s, x, yname, calib, rule, float(a))
    s, x, yname, calib, rule, a = best
    print(f"\nSELECTED (dev macro {s:.3f}): {x}+{yname} | calib={calib} | rule={rule} | alpha={a}")
    At = [cal(p, M[x][calib]) for p in M[x]["test"]]
    Bt = [cal(p, M[yname][calib]) for p in M[yname]["test"]]
    report(At, Bt, M[x]["yt"], rule, a, f"REFINED {x}+{yname} {calib} {rule} a={a}")


if __name__ == "__main__":
    main()
