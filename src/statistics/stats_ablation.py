"""Statistical-test suite: TC-MIL vs the legacy ladder baselines.

Single source of truth for the internal cross-validation tables (tab:cv,
tab:cvstats) and the official-split independent tests (tab:stats). Every legacy
baseline is run at TC-MIL's full budget and aligned run-for-run, so each paired
comparison is matched at every cell:

  K-Fold : 5 repeats x 5 folds x 5 seeds = 125 paired runs (matched on
           (repeat, fold, seed); StratifiedGroupKFold seed = 42 + 1000*rep).
  MC     : 5 splits x 10 seeds           =  50 paired runs (matched on
           (split, seed); StratifiedShuffleSplit over the SORTED UNIQUE subjects
           with seed = 42 -- identical test membership to TC-MIL).

Two regimes:
  PAIRED (same fold/split + seed seen by both models -> matched samples):
      Wilcoxon signed-rank + paired t-test, over the per-run K-Fold / MC metrics
      in results/cross_validation/{kfold_pw1,mc_pw1}/cv_results.json and the
      baselines in results/cross_validation/legacy_{kfold,mc}_aligned/.
  INDEPENDENT (30-seed official per-seed metrics, different seeds):
      Mann-Whitney U + Welch's t-test, over the test_prob_runs in each
      *_official/results.json, recomputed at the a-priori prevalence threshold.

Reads only saved artifacts (no training).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, ttest_ind

from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

TCMIL = {
    "kfold": "results/cross_validation/kfold_pw1/cv_results.json",
    "mc": "results/cross_validation/mc_pw1/cv_results.json",
    "official": "results/single_model/headline_pw1_30seed/results.json",
}
LEGDIR = {
    "kfold": "results/cross_validation/legacy_kfold_aligned/{}_kfold/results.json",
    "mc": "results/cross_validation/legacy_mc_aligned/{}_mc/results.json",
    # Official-split 30-seed baselines (independent regime).
    "official": "results/baselines/legacy_aligned/{}_official/results.json",
}
LADDER = [
    ("dialogue_mean", "dialogue mean"),
    ("flat_mil_mean", "flat MIL, mean"),
    ("flat_mil_attn", "flat MIL, attn"),
    ("damil_r", "DAMIL-R"),
    ("ss_damil_r_mh", "SS-DAMIL-R (MH)"),
    ("ss_damil_r_gsi", "SS-DAMIL-R (GSI)"),
    ("ss_damil_r_conv", "SS-DAMIL-R (Conv)"),
]
METRICS = [
    ("UAR", "balanced_accuracy"), ("Prec.", "precision"), ("Recall", "recall"),
    ("pos-F1", "f1"), ("macro-F1", "macro_f1"), ("micro-F1", "micro_f1"),
    ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"),
]
OFFICIAL_PREV = 0.28  # a-priori (training) prevalence for the official split


def keyed(raw):
    """Key each run by (repeat, group, seed); group is the fold (K-Fold) or the
    split (MC)."""
    out = {}
    for m in raw:
        g = m.get("_fold_idx", m.get("_split_idx"))
        if g is None:
            continue
        out[(m.get("_repeat", 0), g, m["_seed_idx"])] = m
    return out


def boot_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt_p(p):
    if p < 0.001:
        return r"$<$.001$^{*}$"
    s = f"{p:.3f}"
    return s + (r"$^{*}$" if p < 0.05 else "")


def _official_per_seed(path, metric):
    """Per-seed official metric, recomputed from test_prob_runs at the a-priori
    prevalence (testprev) threshold the headline uses -- not the stored
    dev-selected test_per_seed -- so one identical rule is applied to TC-MIL and
    every baseline."""
    r = json.load(open(path))
    y = np.asarray(r["test_labels"])
    out = []
    for prob in r["test_prob_runs"]:
        prob = np.asarray(prob)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=OFFICIAL_PREV)
        out.append(compute_metrics(y, (prob >= t).astype(int), prob)[metric])
    return np.array(out)


def paired_cv(proto):
    """Per-run aggregate (tab:cv) + paired Wilcoxon/paired-t (tab:cvstats)."""
    tc = keyed(json.load(open(TCMIL[proto]))["raw"])
    out = [f"\n########## {proto.upper()}  (TC-MIL cells: {len(tc)}) ##########"]

    out.append(f"\n=== tab:cv  {proto} per-run aggregate "
               f"(mean +/- std [95% CI of mean]) ===")
    rows = []
    for key, name in LADDER:
        p = LEGDIR[proto].format(key)
        if not Path(p).exists():
            out.append(f"  [missing] {p}")
            continue
        rows.append((name, keyed(json.load(open(p))["raw"])))
    rows.append(("TC-MIL", tc))
    for name, kd in rows:
        out.append(f"\n{name}  (n={len(kd)})")
        for label, mk in METRICS:
            vals = np.array([kd[k][mk] for k in kd if mk in kd[k]])
            lo, hi = boot_ci(vals)
            out.append(f"    {label}={vals.mean():.2f}+/-{vals.std(ddof=1):.2f}"
                       f"[{lo:.2f},{hi:.2f}]")

    out.append(f"\n=== tab:cvstats  {proto} paired (Wilcoxon | paired-t) ===")
    for key, name in LADDER:
        p = LEGDIR[proto].format(key)
        if not Path(p).exists():
            continue
        lg = keyed(json.load(open(p))["raw"])
        keys = sorted(set(tc) & set(lg))
        out.append(f"\nTC-MIL vs {name}  (n={len(keys)} matched)")
        for label, mk in METRICS:
            xa = np.array([tc[k][mk] for k in keys])
            xb = np.array([lg[k][mk] for k in keys])
            if np.allclose(xa, xb):
                pw = pt = 1.0
            else:
                _, pw = wilcoxon(xa, xb)
                _, pt = ttest_rel(xa, xb)
            out.append(f"   {label:9s} d={xa.mean()-xb.mean():+.3f}  "
                       f"W={fmt_p(pw)}  t={fmt_p(pt)}")
    return out


def independent_official():
    """Official 30-seed independent tests (tab:stats): Mann-Whitney U + Welch."""
    out = ["\n########## OFFICIAL 30-seed INDEPENDENT (Mann-Whitney | Welch) ##########"]
    if not Path(TCMIL["official"]).exists():
        out.append(f"  [missing] {TCMIL['official']}")
        return out
    for key, name in LADDER:
        p = LEGDIR["official"].format(key)
        if not Path(p).exists():
            out.append(f"\nTC-MIL vs {name}: [missing official baseline {p}]")
            continue
        out.append(f"\nTC-MIL vs {name}")
        for label, mk in METRICS:
            xa = _official_per_seed(TCMIL["official"], mk)
            xb = _official_per_seed(p, mk)
            _, pu = mannwhitneyu(xa, xb, alternative="two-sided")
            _, pt = ttest_ind(xa, xb, equal_var=False)
            out.append(f"   {label:9s} TC={xa.mean():.3f} base={xb.mean():.3f} "
                       f"d={xa.mean()-xb.mean():+.3f}  MWU={fmt_p(pu)}  Welch={fmt_p(pt)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocols", nargs="+", default=["kfold", "mc"],
                    help="paired CV protocols to report")
    ap.add_argument("--independent", action="store_true",
                    help="also report the official 30-seed independent tests")
    ap.add_argument("--out", default="results/stats/ablation_stats.md",
                    help="path to write the report (set empty to skip)")
    args = ap.parse_args()

    lines = []
    for proto in args.protocols:
        lines += paired_cv(proto)
    if args.independent:
        lines += independent_official()

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
