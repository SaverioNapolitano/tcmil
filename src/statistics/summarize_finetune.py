"""Collect every fine-tuning run under results/ft/ into one markdown table.

Run on the cluster (or locally on the synced results/ft/ tree) after the
grid finishes; paste the output into the analysis discussion.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fmt_official(name, r):
    seeds = [m["roc_auc"] for m in r.get("per_seed_dev", [])]
    de = r.get("dev_ensemble", {})
    row = {
        "run": name,
        "method": r["args"].get("ft_method", "?"),
        "dev_ens_auc": de.get("roc_auc"),
        "dev_seed_auc": f"{np.mean(seeds):.4f}±{np.std(seeds):.4f}" if seeds else "",
        "dev_pr_auc": de.get("pr_auc"),
        "_dev_seed_mean": float(np.mean(seeds)) if seeds else None,
        "_dev_seed_std": float(np.std(seeds)) if seeds else None,
    }
    if "test_by_strategy" in r:
        best = r["test_by_strategy"].get("oof_prevalence") \
            or r["test_by_strategy"].get("prevalence")
        row.update({"test_auc": best["roc_auc"], "test_f1": best["f1"],
                    "test_t": best["threshold"]})
    return row


def fmt_cv(name, r):
    ens = r.get("ensemble_aggregate", {})
    return {
        "run": name,
        "method": r["args"].get("ft_method", "?"),
        "ens_auc": ens.get("roc_auc", {}).get("mean"),
        "ens_f1": ens.get("f1", {}).get("mean"),
        "ens_bacc": ens.get("balanced_accuracy", {}).get("mean"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/ft")
    args = p.parse_args()

    official, cv = [], []
    for d in sorted(Path(args.root).glob("**/results.json")):
        r = json.load(open(d))
        name = d.parent.name
        proto = r["args"].get("protocol", "official")
        (official if proto == "official" else cv).append(
            (fmt_official if proto == "official" else fmt_cv)(name, r))

    def table(rows):
        if not rows:
            return "(none)\n"
        keys = [k for k in rows[0].keys() if not k.startswith("_")]
        out = "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n"
        for row in sorted(rows, key=lambda x: -(x.get("dev_ens_auc")
                                                or x.get("ens_auc") or 0)):
            out += "| " + " | ".join(
                f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k])
                for k in keys) + " |\n"
        return out

    print("## Official protocol (sorted by dev ensemble AUC)\n")
    print(table(official))
    print("\n## CV protocols (sorted by ensemble AUC)\n")
    print(table(cv))
    print(select_finalists(official))


def select_finalists(official, band=0.015, cap=4):
    """Pre-registered stage-2 finalist selection (DEV only, see README_FINETUNE).
    Pick every NON-frozen config whose 5-seed per-seed dev AUC mean is within one
    seed-noise band of the best, capped at `cap` to bound test-set multiplicity.
    Frozen control is always carried separately; adoption gate (chosen dev AUC >
    frozen + band) is reported, not enforced here."""
    cands = [r for r in official if r.get("_dev_seed_mean") is not None
             and r["method"] != "frozen"]
    if not cands:
        return "\n## Pre-registered finalists\n\n(no scored non-frozen configs)\n"
    cands.sort(key=lambda r: -r["_dev_seed_mean"])
    best = cands[0]["_dev_seed_mean"]
    band_sel = [r for r in cands if best - r["_dev_seed_mean"] <= band][:cap]
    # The no-adaptation control is the base-encoder frozen run (grid_frozen_ctrl),
    # NOT the DAPT or alt-encoder (mxbai) frozen arms — both also have
    # ft_method=="frozen". Pick it by name; fall back to the best-AUC frozen run
    # (DAPT collapsed, so it can never be the max) so a rename can't grab dapt.
    frozen_cands = [r for r in official if r["method"] == "frozen"
                    and r.get("_dev_seed_mean") is not None]
    frozen = next((r for r in frozen_cands
                   if "dapt" not in r["run"] and "mxbai" not in r["run"]), None) \
        or (max(frozen_cands, key=lambda r: r["_dev_seed_mean"])
            if frozen_cands else None)
    fz = frozen["_dev_seed_mean"] if frozen else None

    out = ["\n## Pre-registered finalists (dev seed-AUC band ±%.3f, cap %d)\n"
           % (band, cap)]
    if len(band_sel) > cap:
        out.append("> NOTE: %d configs tied within band; cap %d applied.\n"
                   % (len([r for r in cands if best - r["_dev_seed_mean"] <= band]), cap))
    for r in band_sel:
        gate = "" if fz is None else (
            " — clears frozen (+%.4f)" % (r["_dev_seed_mean"] - fz)
            if r["_dev_seed_mean"] - fz > band
            else " — does NOT clear frozen by band")
        out.append("- **%s** (%s) dev AUC %.4f±%.4f%s"
                   % (r["run"], r["method"], r["_dev_seed_mean"],
                      r.get("_dev_seed_std") or 0.0, gate))
    if fz is not None:
        out.append("\nfrozen control dev AUC %.4f (adoption needs a finalist > %.4f)."
                   % (fz, fz + band))
    out.append("\nPut these (+ frozen) in cluster/finalists_configs.txt; "
               "set the job-3 array to (#finalists + 1). With >1 finalist, "
               "Holm-correct the FT-vs-frozen test p-values.")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
