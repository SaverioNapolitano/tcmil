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
        keys = list(rows[0].keys())
        out = "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n"
        for row in sorted(rows, key=lambda x: -(x.get("dev_ens_auc")
                                                or x.get("ens_auc") or 0)):
            out += "| " + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in row.values()) + " |\n"
        return out

    print("## Official protocol (sorted by dev ensemble AUC)\n")
    print(table(official))
    print("\n## CV protocols (sorted by ensemble AUC)\n")
    print(table(cv))


if __name__ == "__main__":
    main()
