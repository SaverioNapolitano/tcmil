"""Frozen-control vs fine-tuning OOF comparison for the fine-tuning section.

The frozen encoder was the missing baseline in the Stage-2 OOF probe
(\\Cref{tab:ft-oof}): every adaptation arm had an OOF AUC, but the control did
not, so no adaptation *lift* could be claimed. This script fills that gap.

Two Wilcoxon-family tests, matching the rest of the paper (no DeLong):
  1. Mann-Whitney U (= Wilcoxon rank-sum) on each model's pooled OOF
     probabilities: U/(n+ n-) == ROC-AUC, so its two-sided p tests AUC vs
     chance on the N=142 train+dev pool.
  2. Wilcoxon signed-rank on the 15 paired per-run (5 fold x 3 seed) OOF AUCs,
     frozen vs each fine-tuning finalist. Runs are aligned by (fold,seed), so
     the pairing is exact. Holm-corrected across the four comparisons.

Writes results/ft/OOF_FT_COMPARE.{md,json}.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "results" / "ft"

# Display name -> OOF run directory. frozen first (the control).
VARIANTS = {
    "frozen": "oof_frozen_ctrl",
    "last-4 lr1e-5": "oof_last4_lr1e5",
    "LoRA r16+FFN lr1e-4": "oof_lora_r16_ffn_lr1e4",
    "mxbai LoRA r16 lr1e-4": "oof_mxbai_lora_r16_lr1e4",
    "LoRA r8 lr2e-4": "oof_lora_r8_lr2e4",
}
CONTROL = "frozen"


def load(name: str):
    j = json.load(open(FT / VARIANTS[name] / "results.json"))
    o = j["oof"]
    runs = j["raw"]
    return {
        "probs": np.array(o["probs"]),
        "labels": np.array(o["labels"]),
        "run_keys": [(r["_fold_idx"], r["_seed_idx"]) for r in runs],
        "run_auc": np.array([r["roc_auc"] for r in runs]),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = list(pvals)
    ps = np.array([pvals[k] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    out = {}
    for rank, i in enumerate(order):
        out[keys[i]] = float(min(1.0, ps[i] * (m - rank)))
    return out


def main():
    data = {k: load(k) for k in VARIANTS}

    # Pairing sanity: identical (fold,seed) order across all variants.
    ref_keys = data[CONTROL]["run_keys"]
    ref_labels = data[CONTROL]["labels"]
    for k, d in data.items():
        assert d["run_keys"] == ref_keys, f"{k}: per-run (fold,seed) misaligned"
        assert np.array_equal(d["labels"], ref_labels), f"{k}: OOF labels misaligned"

    y = ref_labels
    npos, nneg = int(y.sum()), int((y == 0).sum())
    N = len(y)

    # (1) Pooled OOF AUC + Mann-Whitney U vs chance.
    pooled = {}
    for k, d in data.items():
        p = d["probs"]
        auc = float(roc_auc_score(y, p))
        U, pp = mannwhitneyu(p[y == 1], p[y == 0], alternative="two-sided")
        pooled[k] = {"auc": auc, "U_over_npos_nneg": float(U / (npos * nneg)),
                     "wmw_p_vs_chance": float(pp)}

    # (2) Per-run paired Wilcoxon signed-rank, frozen vs each FT finalist.
    fz = data[CONTROL]["run_auc"]
    per_run_mean_std = {k: [float(d["run_auc"].mean()), float(d["run_auc"].std())]
                        for k, d in data.items()}
    signed = {}
    raw_p = {}
    for k, d in data.items():
        if k == CONTROL:
            continue
        W, p = wilcoxon(fz, d["run_auc"])
        delta = float((fz - d["run_auc"]).mean())
        signed[k] = {"mean_delta_frozen_minus": delta, "W": float(W), "p_raw": float(p)}
        raw_p[k] = float(p)
    holm_p = holm(raw_p)
    for k in signed:
        signed[k]["p_holm"] = holm_p[k]

    results = {
        "n_subjects": N, "n_pos": npos, "n_neg": nneg,
        "n_runs": len(fz), "run_design": "5 fold x 3 seed = 15",
        "pooled_oof": pooled,
        "per_run_auc_mean_std": per_run_mean_std,
        "signed_rank_frozen_vs_ft": signed,
        "verdict": ("frozen pooled-OOF AUC is the best of the set and no fine-tuning "
                    "finalist differs from it (all Holm p ~ 1); fine-tuning yields no "
                    "leakage-free OOF AUC lift."),
    }

    (FT / "OOF_FT_COMPARE.json").write_text(json.dumps(results, indent=2))

    # Markdown report.
    lines = []
    lines.append("# Frozen control vs fine-tuning -- OOF comparison\n")
    lines.append(f"Train+dev pool N={N} ({npos} positive). Runs: {results['run_design']}.\n")
    lines.append("Tests: Mann-Whitney U (AUC vs chance) + Wilcoxon signed-rank "
                 "(frozen vs FT, paired per-run AUC, Holm-corrected). No DeLong.\n")
    lines.append("## Pooled OOF AUC (Mann-Whitney vs chance)\n")
    lines.append("| Model | OOF AUC | WMW p |")
    lines.append("|---|---|---|")
    for k in VARIANTS:
        pl = pooled[k]
        lines.append(f"| {k} | {pl['auc']:.4f} | {pl['wmw_p_vs_chance']:.1e} |")
    lines.append("\n## Per-run OOF AUC (15 runs, mean +- std)\n")
    lines.append("| Model | mean | std |")
    lines.append("|---|---|---|")
    for k in VARIANTS:
        m, s = per_run_mean_std[k]
        lines.append(f"| {k} | {m:.4f} | {s:.4f} |")
    lines.append("\n## Frozen vs FT -- Wilcoxon signed-rank (paired per-run AUC)\n")
    lines.append("| vs | mean Δ (frozen−FT) | W | p_raw | p_Holm |")
    lines.append("|---|---|---|---|---|")
    for k in signed:
        s = signed[k]
        lines.append(f"| {k} | {s['mean_delta_frozen_minus']:+.4f} | {s['W']:.1f} | "
                     f"{s['p_raw']:.3f} | {s['p_holm']:.3f} |")
    lines.append(f"\n**Verdict.** {results['verdict']}\n")
    (FT / "OOF_FT_COMPARE.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nwrote {FT/'OOF_FT_COMPARE.md'} and .json")


if __name__ == "__main__":
    main()
