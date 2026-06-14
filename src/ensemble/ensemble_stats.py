"""Full statistical rigor for the FINAL ensemble (parity with the single model).

Final ensemble = bge-large GRU + UAE-large GRU, per-member temperature scaling,
arithmetic mean, a-priori prevalence threshold (the recipe selected leakage-free
on dev; all diversity variants were rejected, see ENSEMBLE-ROADMAP.md).

Reports, mirroring tab:external / SEED_STABILITY for the single model:
  1. per-seed mean +/- std (full 30-seed and 6x(5+5)) for macro/micro/AUC,
  2. subject-level bootstrap 95% CI (2000 resamples of the 47 test subjects) on
     the seed-averaged ensemble at the prevalence threshold,
  3. one-sample t vs Milintsevich 0.739 and vs the single bar 0.774 (macro+micro),
  4. ensemble-vs-single significance: paired (seed-aligned, base_seed 100) and
     independent tests on the 30 per-seed macro-F1 values.
Writes results/ensemble/ENSEMBLE_STATS.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV, MIL, BAR = 0.28, 0.739, 0.774
EPS = 1e-6
A = "results/ensemble/pw1_members/bge_gru_pw1/results.json"
B = "results/ensemble/pw1_members/uae_gru_pw1/results.json"
SINGLE = "results/single_model/headline_pw1_30seed/results.json"


def _lg(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def _sg(z): return 1 / (1 + np.exp(-z))


def temp(dev, y):
    z = [_lg(np.asarray(p)) for p in dev]; y = np.asarray(y, float)
    def nll(T):
        T = max(T, 1e-3); t = 0.0
        for zi in z:
            p = np.clip(_sg(zi / T), EPS, 1 - EPS)
            t += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return t / len(z)
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def per_seed_macro(runs, y, key="macro_f1"):
    out = []
    for r in runs:
        t = tune_threshold(None, r, metric="prevalence", prevalence=PREV)
        out.append(compute_metrics(y, (r >= t).astype(int), r)[key])
    return np.array(out)


def boot_ci(prob, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
    pred = (prob >= t).astype(int)
    keys = ("macro_f1", "micro_f1", "f1", "roc_auc")
    boots = {k: [] for k in keys}
    N = len(y)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], pred[idx], prob[idx])
        for k in keys:
            boots[k].append(m[k])
    point = compute_metrics(y, pred, prob)
    return {k: (point[k], np.percentile(boots[k], 2.5), np.percentile(boots[k], 97.5)) for k in keys}


def main():
    ra, rb, rs = json.load(open(A)), json.load(open(B)), json.load(open(SINGLE))
    y = np.asarray(ra["test_labels"])
    Ta = temp(ra["dev_prob_runs"], ra["dev_labels"])
    Tb = temp(rb["dev_prob_runs"], rb["dev_labels"])
    ens = [( _sg(_lg(np.asarray(ra["test_prob_runs"][i])) / Ta)
            + _sg(_lg(np.asarray(rb["test_prob_runs"][i])) / Tb)) / 2
           for i in range(min(len(ra["test_prob_runs"]), len(rb["test_prob_runs"])))]
    n = len(ens)
    single = [np.asarray(p) for p in rs["test_prob_runs"]]

    L = ["# Ensemble statistics — bge-gru + uae-gru (temp-scaled, prevalence)\n"]

    # 1. per-seed mean+/-std
    L.append("## Per-seed (mean +/- std)\n")
    L.append("| protocol | macro-F1 | micro-F1 | ROC-AUC |")
    L.append("|---|---|---|---|")
    full = {k: per_seed_macro(ens, y, k) for k in ("macro_f1", "micro_f1", "roc_auc")}
    L.append(f"| full {n}-seed | {full['macro_f1'].mean():.3f}±{full['macro_f1'].std():.3f} "
             f"| {full['micro_f1'].mean():.3f}±{full['micro_f1'].std():.3f} "
             f"| {full['roc_auc'].mean():.3f}±{full['roc_auc'].std():.3f} |")
    g = n // 5
    grp = {k: [] for k in ("macro_f1", "micro_f1", "roc_auc")}
    for gi in range(g):
        prob = np.mean(ens[gi*5:gi*5+5], axis=0)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
        m = compute_metrics(y, (prob >= t).astype(int), prob)
        for k in grp: grp[k].append(m[k])
    grp = {k: np.array(v) for k, v in grp.items()}
    L.append(f"| {g}x(5+5) | {grp['macro_f1'].mean():.3f}±{grp['macro_f1'].std():.3f} "
             f"| {grp['micro_f1'].mean():.3f}±{grp['micro_f1'].std():.3f} "
             f"| {grp['roc_auc'].mean():.3f}±{grp['roc_auc'].std():.3f} |")

    # 2. subject-level bootstrap CI on seed-averaged ensemble
    L.append(f"\n## Subject-level bootstrap 95% CI (2000 resamples, N={len(y)})\n")
    ci = boot_ci(np.mean(ens, axis=0), y)
    L.append("| metric | point | 95% CI |")
    L.append("|---|---|---|")
    for k in ("macro_f1", "micro_f1", "roc_auc"):
        p, lo, hi = ci[k]
        L.append(f"| {k} | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")

    # 3. one-sample t vs baselines (macro + micro)
    L.append("\n## One-sample t (per-seed, full 30)\n")
    L.append("| metric | vs Milintsevich 0.739 | vs single 0.774 |")
    L.append("|---|---|---|")
    for k, ref2 in [("macro_f1", BAR), ("micro_f1", None)]:
        v = full[k]
        pm = stats.ttest_1samp(v, MIL)[1]
        s = f"| {k} | Δ={v.mean()-MIL:+.3f} p={pm:.3g} |"
        if ref2:
            ps = stats.ttest_1samp(v, ref2)[1]
            s += f" Δ={v.mean()-ref2:+.3f} p={ps:.3g} |"
        else:
            s += " — |"
        L.append(s)

    # 4. ensemble vs single (30 seeds each, seed-aligned base_seed 100)
    L.append("\n## Ensemble vs single model (per-seed macro-F1, 30 seeds)\n")
    se = full["macro_f1"]; ss = per_seed_macro(single[:n], y)
    paired_t = stats.ttest_rel(se, ss)[1]
    wilc = stats.wilcoxon(se, ss)[1]
    mwu = stats.mannwhitneyu(se, ss, alternative="two-sided")[1]
    welch = stats.ttest_ind(se, ss, equal_var=False)[1]
    L.append(f"- ensemble {se.mean():.3f}±{se.std():.3f}  vs  single {ss.mean():.3f}±{ss.std():.3f}  "
             f"(Δ={se.mean()-ss.mean():+.3f})")
    L.append(f"- paired t p={paired_t:.3g} ; Wilcoxon p={wilc:.3g} ; "
             f"Mann-Whitney p={mwu:.3g} ; Welch t p={welch:.3g}")

    out = Path("results/ensemble/ENSEMBLE_STATS.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
