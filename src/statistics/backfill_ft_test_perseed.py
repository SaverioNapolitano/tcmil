"""Backfill per-seed test predictions/metrics into FT official results.json.

The official fine-tuning runs (results/ft/official_*) persisted only the
30-seed *ensemble* test vector in results.json, but every per-seed test
prediction is already in seed_cache/seed_*.json. This script re-aggregates
those caches -- NO retraining -- to add:

    test_prob_runs : list of per-seed (47,) test probability vectors
    test_per_seed  : per-seed test metrics at the run's prevalence threshold
    test_per_seed_summary : mean+/-std of per-seed roc_auc and macro_f1

so the official-test table can report a per-seed mean+/-std AUC matching
tab:external / tab:levers. This realises option "B" at zero cluster cost;
future runs save these keys natively (see finetune_tcmil.py).

    python -m src.statistics.backfill_ft_test_perseed            # all official_*
    python -m src.statistics.backfill_ft_test_perseed --dry_run  # report only
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _threshold(results):
    ts = results.get("test_by_strategy", {})
    blk = ts.get("oof_prevalence") or ts.get("prevalence")
    return float(blk["threshold"]) if blk else 0.5


def _per_seed(cache_dir, thr):
    probs, mets = [], []
    for f in sorted(glob.glob(str(cache_dir / "seed_*.json"))):
        u = json.load(open(f))
        u = u.get("data", u)
        if "test_probs" not in u:
            continue
        y = np.array(u["test_labels"])
        p = np.array(u["test_probs"])
        pred = (p >= thr).astype(int)
        probs.append(p.tolist())
        mets.append({
            "roc_auc": float(roc_auc_score(y, p)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        })
    return probs, mets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ft")
    ap.add_argument("--glob", default="official_*")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    for d in sorted(Path(args.root).glob(args.glob)):
        rj, cd = d / "results.json", d / "seed_cache"
        if not rj.exists() or not cd.is_dir():
            continue
        R = json.load(open(rj))
        if "test_by_strategy" not in R:  # not an eval_test run
            continue
        thr = _threshold(R)
        probs, mets = _per_seed(cd, thr)
        if not mets:
            print(f"{d.name}: no cached per-seed test probs, skip")
            continue
        aucs = [m["roc_auc"] for m in mets]
        mfs = [m["macro_f1"] for m in mets]
        summary = {
            "n_seeds": len(mets), "threshold": thr,
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "macro_f1_mean": float(np.mean(mfs)), "macro_f1_std": float(np.std(mfs)),
        }
        print(f"{d.name:32s} n={len(mets):2d}  AUC {summary['roc_auc_mean']:.3f}"
              f"+/-{summary['roc_auc_std']:.3f}  macroF1 {summary['macro_f1_mean']:.3f}"
              f"+/-{summary['macro_f1_std']:.3f}  (thr {thr:.2f})")
        if args.dry_run:
            continue
        R["test_prob_runs"] = probs
        R["test_per_seed"] = mets
        R["test_per_seed_summary"] = summary
        json.dump(R, open(rj, "w"), indent=2)


if __name__ == "__main__":
    main()
