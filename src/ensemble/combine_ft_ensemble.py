"""Combine fine-tuned (FT) members into an offline ensemble.

The FT analogue of `combine_oof_ensemble.py`. Each FT finalist produces two
runs (see `cluster/run_ft_finalists.sh`):

    oof_<name>/results.json       --protocol export_oof : pooled OOF probs +
                                    leakage-free thresholds (results["oof"]).
    official_<name>/results.json  --protocol official --eval_test : test_probs.

A *member* therefore is the pair (oof_dir, official_dir). The ensemble averages
probabilities at both stages, exactly like the frozen OOF ensemble:

    1. average OOF probs  -> tune the decision threshold (per strategy)
    2. average test probs -> evaluate once at those thresholds

OOF/test probabilities are aligned across members by construction (same pool,
same split seed, same test load order), so plain averaging is valid. No
re-training is needed — this consumes only the saved probabilities.

Usage:
    python src/ensemble/combine_ft_ensemble.py \
        --member results/ft/oof_winner1 results/ft/official_winner1 \
        --member results/ft/oof_winner2 results/ft/official_winner2 \
        --output results/ft/ft_ensemble/results.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics


def load_member(oof_dir: Path, official_dir: Path):
    oof = json.load(open(oof_dir / "results.json"))["oof"]
    off = json.load(open(official_dir / "results.json"))
    oof_labels = np.array(oof["labels"])
    return {
        "name": official_dir.name.replace("official_", ""),
        "oof_probs": np.array(oof["probs"]),
        "oof_labels": oof_labels,
        "test_probs": np.array(off["test_probs"]),
        "test_labels": np.array(off["test_labels"]),
        # export_oof does not persist pool prevalence; the OOF set covers the
        # whole pool so its label mean is the prevalence.
        "pool_prev": float(oof_labels.mean()),
    }


def evaluate(members, label):
    oof_labels = members[0]["oof_labels"]
    test_labels = members[0]["test_labels"]
    for m in members[1:]:
        assert np.array_equal(m["oof_labels"], oof_labels), "OOF label misalignment"
        assert np.array_equal(m["test_labels"], test_labels), "test label misalignment"

    oof_avg = np.mean([m["oof_probs"] for m in members], axis=0)
    test_avg = np.mean([m["test_probs"] for m in members], axis=0)
    pool_prev = members[0]["pool_prev"]

    oof_auc = compute_metrics(oof_labels, (oof_avg >= 0.5).astype(int), oof_avg)["roc_auc"]
    out = {"members": [m["name"] for m in members], "oof_auc": float(oof_auc),
           "test_by_strategy": {}}
    print(f"\n--- {label} (OOF AUC {oof_auc:.4f}) ---")
    for metric in ["f1", "bacc", "youden", "prevalence"]:
        t = tune_threshold(oof_labels, oof_avg, metric, prevalence=pool_prev)
        tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        out["test_by_strategy"][metric] = {"threshold": float(t), **tm}
        print(f"  [OOF-{metric:<10}] t={t:.2f}  F1={tm['f1']:.4f} P={tm['precision']:.4f} "
              f"R={tm['recall']:.4f} macroF1={tm['macro_f1']:.4f} AUC={tm['roc_auc']:.4f}")
    return out


def main():
    p = argparse.ArgumentParser(description="Fine-tuned multi-member OOF ensemble")
    p.add_argument("--member", nargs=2, action="append", required=True,
                   metavar=("OOF_DIR", "OFFICIAL_DIR"),
                   help="A member = its export_oof dir AND its official-test dir. "
                        "Repeat --member for each ensemble member.")
    p.add_argument("--output", default="results/ft/ft_ensemble/results.json")
    args = p.parse_args()

    members = [load_member(Path(o), Path(off)) for o, off in args.member]
    results = {"singles": {}, "combos": {}}

    for m in members:
        results["singles"][m["name"]] = evaluate([m], m["name"])
    for k in range(2, len(members) + 1):
        for combo in combinations(members, k):
            label = "+".join(c["name"] for c in combo)
            results["combos"][label] = evaluate(list(combo), label)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
