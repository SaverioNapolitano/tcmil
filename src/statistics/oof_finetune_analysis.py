"""Leakage-free OOF analysis of the Stage-2 fine-tuning finalists.

Stage 2 of the fine-tuning study (see README) gives every finalist a
nested out-of-fold (OOF) probe over the train+dev pool (one leakage-free
probability per subject) to fix the decision threshold. That same probe is a
leakage-free, larger-N (N=142 vs. dev N=35) estimate of how the finalists rank
*before* the once-only test evaluation (jobs 3/4, pending). This script reads
those OOF probabilities (the `oof` block of each finalist's results.json) and:

  1. Per finalist: OOF ROC-AUC and PR-AUC with a subject-resampling bootstrap
     95% CI (same resampling as stats_tcmil.bootstrap_official).
  2. Pairwise: paired bootstrap of the OOF ROC-AUC difference vs. the OOF-best
     finalist (shared resample indices -> paired), with a two-sided bootstrap
     p-value. Tells us whether the OOF ranking is real or seed/sample noise.
  3. Dev (per-seed mean) rank vs. OOF rank, to show whether the development
     grid ranking survives the leakage-free probe.

This touches NO test data; it only re-reads probabilities already produced by
the Stage-2 OOF probe. Run after the finalist OOF jobs land:

    python src/statistics/oof_finetune_analysis.py --root results/ft
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent.parent))


def _dev_seed_auc(grid_dir):
    """Per-seed mean dev ROC-AUC for the matching Stage-1 grid run, or None."""
    f = grid_dir / "results.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    seeds = [m["roc_auc"] for m in r.get("per_seed_dev", [])]
    return float(np.mean(seeds)) if seeds else None


def load_finalists(root):
    """Return {name: (labels, probs, dev_seed_auc)} for every oof_* finalist.

    Asserts all finalists share the same OOF subject set and label vector, so
    the bootstrap below can pair resample indices across configs.
    """
    out, ref_labels = {}, None
    for d in sorted(root.glob("oof_*")):
        rj = d / "results.json"
        if not rj.exists():
            continue
        oof = json.load(open(rj)).get("oof")
        if oof is None:
            continue
        labels = np.array(oof["labels"])
        probs = np.array(oof["probs"])
        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(ref_labels, labels):
            raise ValueError(
                f"{d.name} OOF labels are not aligned with the other finalists; "
                "paired bootstrap requires a shared subject set/order.")
        name = d.name[len("oof_"):]
        grid = root / ("grid_" + name)
        out[name] = (labels, probs, _dev_seed_auc(grid))
    if not out:
        raise SystemExit(f"no oof_*/results.json with an 'oof' block under {root}")
    return out


def analyse(finalists, n_boot=5000, seed=42):
    names = list(finalists)
    y = finalists[names[0]][0]
    rng = np.random.default_rng(seed)
    # Shared resample indices -> paired comparison across configs. Drop draws
    # with a single class (AUC undefined), exactly like bootstrap_official.
    idx = [i for i in (rng.choice(len(y), len(y), replace=True) for _ in range(n_boot))
           if len(np.unique(y[i])) > 1]

    rows = {}
    for name, (_, probs, dev_auc) in finalists.items():
        bs = np.array([roc_auc_score(y[i], probs[i]) for i in idx])
        rows[name] = {
            "auc": roc_auc_score(y, probs),
            "pr_auc": average_precision_score(y, probs),
            "ci": (np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
            "boot": bs,
            "dev_auc": dev_auc,
        }

    order = sorted(rows, key=lambda n: -rows[n]["auc"])
    best = order[0]

    print(f"## Stage-2 OOF probe — finalist comparison (leakage-free, "
          f"N={len(y)}, pos={int(y.sum())}, {len(idx)} bootstraps)\n")
    print("| finalist | OOF AUC | 95% CI | OOF PR-AUC | dev seed-AUC | dev rank | OOF rank |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    dev_order = sorted([n for n in rows if rows[n]["dev_auc"] is not None],
                       key=lambda n: -rows[n]["dev_auc"])
    dev_rank = {n: i + 1 for i, n in enumerate(dev_order)}
    for i, n in enumerate(order, 1):
        r = rows[n]
        dv = f"{r['dev_auc']:.4f}" if r["dev_auc"] is not None else "—"
        dr = str(dev_rank.get(n, "—"))
        star = " **(OOF-best)**" if n == best else ""
        print(f"| {n}{star} | {r['auc']:.4f} | "
              f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | {r['pr_auc']:.4f} | "
              f"{dv} | {dr} | {i} |")

    print(f"\n## Paired ΔAUC vs OOF-best ({best}) — same resampled subjects\n")
    print("| comparison | ΔAUC | 95% CI | p (two-sided) |")
    print("|---|---:|---:|---:|")
    base = rows[best]["boot"]
    any_sig = False
    for n in order[1:]:
        d = base - rows[n]["boot"]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        any_sig |= p < 0.05
        print(f"| {best} − {n} | {d.mean():+.4f} | "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] | {p:.3f} |")

    print()
    if not any_sig:
        print("**Verdict:** no pairwise OOF AUC difference is significant "
              "(every CI spans 0). The finalists are statistically "
              "indistinguishable on the leakage-free probe; the development-grid "
              "ranking does not survive it. The once-only test split arbitrates.")
    else:
        print("**Verdict:** at least one finalist separates on the leakage-free "
              "OOF probe (see significant rows above).")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/ft", type=Path)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    analyse(load_finalists(args.root), n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
