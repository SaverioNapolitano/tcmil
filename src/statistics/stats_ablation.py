"""Full statistical test suite: TC-MIL vs each legacy baseline.

Reads only saved artifacts (no training). Two regimes:

PAIRED (same fold + seed seen by both models -> matched samples):
    - Wilcoxon signed-rank (non-parametric)
    - paired t-test (parametric)
  Source: per-run K-Fold and MC metrics in results/cross_validation/<tcmil>/cv_results.json
  and results/baselines/legacy_aligned/<model>_<kfold|mc>/results.json, joined on
  (_repeat, _fold_idx, _seed_idx). The CV split seed is shared (random_state
  42), so fold membership is identical across models -> legitimately paired.

INDEPENDENT (per-seed official metrics, different seeds -> independent samples):
    - Mann-Whitney U (non-parametric)
    - Welch's t-test (parametric, unequal variance)
  Source: test_per_seed (30 seeds) in each *_official/results.json.

Metrics tested: macro_f1 and roc_auc.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind

sys.path.append(str(Path(__file__).parent.parent.parent))

# Best single model = bge-large + GRU + pos_weight=1.0 (the winning recipe).
TCMIL = {
    "kfold": "results/cross_validation/kfold_pw1/cv_results.json",
    "mc": "results/cross_validation/mc_pw1/cv_results.json",
    "official": "results/single_model/headline_pw1_30seed/results.json",
}
LEGACY = ["dialogue_mean", "flat_mil_mean", "flat_mil_attn", "damil_r",
          "ss_damil_r_v9", "ss_damil_r_v25", "ss_damil_r_v26", "ss_damil_r_v29"]


from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

# A-priori prevalence threshold rates (same used to produce the stored
# macro-F1): train prevalence for the official split, CV-pool prevalence for
# K-Fold/MC. micro-F1 (≡ accuracy) is recomputed at this fixed rate, so it is
# honest, not a test-tuned oracle.
OFFICIAL_PREV = 0.28


def keyed(raw):
    out = {}
    for m in raw:
        group = m.get("_fold_idx", m.get("_split_idx"))
        out[(m.get("_repeat", 0), group, m["_seed_idx"])] = m
    return out


def _metric_from_run(run, metric, prev):
    """Return run[metric] if present; else recompute (micro_f1) from the run's
    saved probabilities at the prevalence-matched threshold."""
    if metric in run and run[metric] is not None:
        return run[metric]
    prob = np.asarray(run["probability"]); y = np.asarray(run["true_label"])
    t = tune_threshold(None, prob, metric="prevalence", prevalence=prev)
    return compute_metrics(y, (prob >= t).astype(int), prob)[metric]


def _pool_prev(raw):
    """CV-pool prevalence = mean label over one repeat's concatenated folds."""
    rep0 = [m for m in raw if m.get("_repeat", 0) == 0 and m.get("_seed_idx") == 0]
    ys = np.concatenate([np.asarray(m["true_label"]) for m in rep0]) if rep0 else None
    return float(ys.mean()) if ys is not None and len(ys) else 0.30


def paired(tcmil_raw, legacy_path, metric):
    if not Path(legacy_path).exists():
        return None
    a = keyed(tcmil_raw)
    b = keyed(json.load(open(legacy_path))["raw"])
    keys = sorted(set(a) & set(b))
    if len(keys) < 6:
        return None
    pa, pb = _pool_prev(tcmil_raw), _pool_prev(json.load(open(legacy_path))["raw"])
    xa = np.array([_metric_from_run(a[k], metric, pa) for k in keys])
    xb = np.array([_metric_from_run(b[k], metric, pb) for k in keys])
    w, pw = wilcoxon(xa, xb)
    t, pt = ttest_rel(xa, xb)
    return dict(n=len(keys), a=xa.mean(), b=xb.mean(), d=xa.mean() - xb.mean(),
                wilcoxon_p=pw, paired_t_p=pt)


def _official_per_seed(path, metric):
    """Per-seed official metric, recomputed from test_prob_runs at the **same
    a-priori prevalence (testprev) threshold** the headline uses — NOT the
    stored test_per_seed (which used a dev-selected threshold). This keeps the
    stats consistent with the reported headline and applies one identical
    threshold rule to TC-MIL and every legacy model."""
    r = json.load(open(path))
    y = np.asarray(r["test_labels"])
    out = []
    for prob in r["test_prob_runs"]:
        prob = np.asarray(prob)
        t = tune_threshold(None, prob, metric="prevalence", prevalence=OFFICIAL_PREV)
        out.append(compute_metrics(y, (prob >= t).astype(int), prob)[metric])
    return np.array(out)


def independent(tcmil_official, legacy_official, metric):
    if not Path(legacy_official).exists():
        return None
    xa = _official_per_seed(tcmil_official, metric)
    xb = _official_per_seed(legacy_official, metric)
    u, pu = mannwhitneyu(xa, xb, alternative="two-sided")
    t, pt = ttest_ind(xa, xb, equal_var=False)
    return dict(na=len(xa), nb=len(xb), a=xa.mean(), b=xb.mean(),
                d=xa.mean() - xb.mean(), mwu_p=pu, welch_t_p=pt)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stats/ablation_stats.md")
    args = ap.parse_args()
    lines = ["# Statistical tests — TC-MIL vs legacy baselines\n",
             "TC-MIL: K-Fold = bge-large+GRU (5x5 repeats); MC = 3-member s10; "
             "official = bge-large 30-seed. Legacy = `results/baselines/legacy_aligned/`. "
             "Metrics: macro-F1, micro-F1 (≡accuracy), ROC-AUC. micro-F1 is "
             "recomputed from saved probabilities at the same a-priori "
             "prevalence threshold as macro-F1 (honest, not test-tuned). "
             "p-values two-sided; `***`<0.001 `**`<0.01 `*`<0.05.\n"]

    kf = json.load(open(TCMIL["kfold"]))["raw"]
    mc = json.load(open(TCMIL["mc"]))["raw"]

    for metric in ["macro_f1", "micro_f1", "roc_auc"]:
        lines.append(f"\n## {metric} — PAIRED (matched fold+seed)\n")
        lines.append("| vs | proto | n | TC-MIL | legacy | Δ | Wilcoxon p | paired-t p |")
        lines.append("|---|---|--:|--:|--:|--:|--:|--:|")
        for model in LEGACY:
            for proto, raw in [("kfold", kf), ("mc", mc)]:
                r = paired(raw, f"results/baselines/legacy_aligned/{model}_{proto}/results.json", metric)
                if r:
                    lines.append(f"| {model} | {proto} | {r['n']} | {r['a']:.3f} | "
                                 f"{r['b']:.3f} | {r['d']:+.3f} | {r['wilcoxon_p']:.3g} "
                                 f"{stars(r['wilcoxon_p'])} | {r['paired_t_p']:.3g} "
                                 f"{stars(r['paired_t_p'])} |")

        lines.append(f"\n## {metric} — INDEPENDENT (30-seed official per-seed)\n")
        lines.append("| vs | nA/nB | TC-MIL | legacy | Δ | Mann-Whitney U p | Welch-t p |")
        lines.append("|---|---|--:|--:|--:|--:|--:|")
        for model in LEGACY:
            r = independent(TCMIL["official"],
                            f"results/baselines/legacy_aligned/{model}_official/results.json", metric)
            if r:
                lines.append(f"| {model} | {r['na']}/{r['nb']} | {r['a']:.3f} | "
                             f"{r['b']:.3f} | {r['d']:+.3f} | {r['mwu_p']:.3g} "
                             f"{stars(r['mwu_p'])} | {r['welch_t_p']:.3g} {stars(r['welch_t_p'])} |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
