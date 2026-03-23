#!/usr/bin/env python3
"""
Sanity-check baselines for binary interview-level classification.

Baselines implemented:
  1. Majority-class  — always predict the most frequent training label.
  2. Stratified random — sample predictions from the training class prior.

Usage:
    # Simple format (train.csv / val.csv / test.csv with columns interview_id, label):
    python scripts/run_baselines.py --data-dir data/splits --output-dir outputs/baselines

    # DAIC-WOZ format (auto-detected):
    python scripts/run_baselines.py --data-dir data/labels --output-dir outputs/baselines

Prerequisites:
    pip install numpy pandas scikit-learn matplotlib seaborn

Input format (auto-detected):
    Option A — Simple format:
        data-dir/ must contain train.csv, val.csv, test.csv
        Each CSV has columns: interview_id, label  (label ∈ {0, 1})
    Option B — DAIC-WOZ / AVEC2017 format:
        data-dir/ contains train_split_Depression_AVEC2017.csv,
        dev_split_Depression_AVEC2017.csv, full_test_split.csv
        ID column: Participant_ID;  label column: PHQ8_Binary or PHQ_Binary

Outputs (written to output-dir/):
    metrics.json              — machine-readable metrics
    summary.csv               — one row per (split, baseline)
    stratified_per_seed.csv   — one row per seed × split
    report.md                 — human-readable markdown report
    class_distribution.png    — bar chart of label counts
    majority_confusion_*.png  — confusion matrices for majority baseline
    stratified_boxplot_*.png  — metric boxplots across seeds
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_utils import (
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metric_boxplot,
)


# ────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────

# Maps split names to file-name candidates (tried in order)
_SPLIT_FILE_CANDIDATES = {
    "train": ["train.csv", "train_split_Depression_AVEC2017.csv"],
    "val":   ["val.csv",   "dev_split_Depression_AVEC2017.csv"],
    "test":  ["test.csv",  "full_test_split.csv"],
}

# Possible column names for interview ID and binary label
_ID_COLS    = ["interview_id", "Participant_ID"]
_LABEL_COLS = ["label", "PHQ8_Binary", "PHQ_Binary"]


def _find_column(df, candidates, path):
    """Return the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of {candidates} found in {path}. Columns: {list(df.columns)}"
    )


def load_splits(data_dir):
    """Load train/val/test CSVs, auto-detecting format.

    Supports both the simple format (interview_id, label) and the DAIC-WOZ
    format (Participant_ID, PHQ8_Binary / PHQ_Binary).  Normalizes columns to
    'interview_id' and 'label' so downstream code is format-agnostic.
    """
    splits = {}
    for split_name, candidates in _SPLIT_FILE_CANDIDATES.items():
        # Try each candidate filename
        path = None
        for fname in candidates:
            p = os.path.join(data_dir, fname)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"No file found for '{split_name}' split in {data_dir}. "
                f"Tried: {candidates}"
            )

        df = pd.read_csv(path)

        # Normalize column names
        id_col = _find_column(df, _ID_COLS, path)
        label_col = _find_column(df, _LABEL_COLS, path)
        df = df.rename(columns={id_col: "interview_id", label_col: "label"})

        splits[split_name] = df[["interview_id", "label"]]
        print(f"Loaded {split_name}: {len(df)} samples  (from {os.path.basename(path)})")

    return splits


# ────────────────────────────────────────────────────────
# Majority baseline
# ────────────────────────────────────────────────────────

def run_majority_baseline(train_labels, eval_labels):
    """Predict the training majority class for every eval sample.

    Also returns a constant probability = training positive-class rate,
    so ROC AUC / PR AUC can be computed.
    """
    majority_label = int(pd.Series(train_labels).mode()[0])
    pos_rate = float(np.mean(train_labels))

    y_pred = np.full(len(eval_labels), majority_label)
    y_prob = np.full(len(eval_labels), pos_rate)

    return y_pred, y_prob, majority_label, pos_rate


# ────────────────────────────────────────────────────────
# Stratified random baseline
# ────────────────────────────────────────────────────────

def run_stratified_baseline(train_labels, eval_labels, seed):
    """Sample predictions from the training class prior."""
    pos_rate = float(np.mean(train_labels))
    rng = np.random.default_rng(seed)
    y_pred = (rng.random(len(eval_labels)) < pos_rate).astype(int)
    # Use the prior as predicted probability for every sample
    y_prob = np.full(len(eval_labels), pos_rate)
    return y_pred, y_prob


# ────────────────────────────────────────────────────────
# Report generation
# ────────────────────────────────────────────────────────

def format_metrics_block(metrics, indent=""):
    """Pretty-print a metrics dict as text lines."""
    lines = []
    for k, v in metrics.items():
        if k == "confusion_matrix":
            lines.append(f"{indent}{k}:")
            lines.append(f"{indent}  {v[0]}")
            lines.append(f"{indent}  {v[1]}")
        elif v is None:
            lines.append(f"{indent}{k}: N/A")
        else:
            lines.append(f"{indent}{k}: {v:.4f}")
    return "\n".join(lines)


def build_markdown_report(splits, train_dist, majority_results, stratified_agg, output_dir):
    """Create a human-readable markdown report."""
    lines = [
        "# Baseline Evaluation Report",
        "",
        "## Training Class Distribution",
        "",
        f"| Label | Count | Rate |",
        f"|-------|-------|------|",
    ]
    for label_val in sorted(train_dist.keys()):
        count = train_dist[label_val]
        total = sum(train_dist.values())
        lines.append(f"| {label_val} | {count} | {count/total:.3f} |")

    lines += [
        "",
        "## Baseline Definitions",
        "",
        "- **Majority baseline**: always predict the most frequent training label."
        " A constant probability equal to the training positive-class rate is used"
        " for ROC AUC / PR AUC computation.",
        "- **Stratified random baseline**: sample predictions according to the"
        " training class prior. Repeated over multiple seeds; mean ± std reported.",
        "",
        "## Majority Baseline Results",
        "",
    ]

    for split_name in ["val", "test"]:
        m = majority_results[split_name]
        lines.append(f"### {split_name.capitalize()}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in m.items():
            if k == "confusion_matrix":
                continue
            val_str = f"{v:.4f}" if v is not None else "N/A"
            lines.append(f"| {k} | {val_str} |")
        cm = m["confusion_matrix"]
        lines.append("")
        lines.append("Confusion matrix (rows=true, cols=pred):")
        lines.append("")
        lines.append("```")
        lines.append(f"  Pred 0  Pred 1")
        lines.append(f"True 0  {cm[0][0]:5d}  {cm[0][1]:5d}")
        lines.append(f"True 1  {cm[1][0]:5d}  {cm[1][1]:5d}")
        lines.append("```")
        lines.append("")

    lines.append("## Stratified Random Baseline Results (aggregated)")
    lines.append("")

    for split_name in ["val", "test"]:
        agg = stratified_agg[split_name]
        lines.append(f"### {split_name.capitalize()}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for metric_name in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            mean_val = agg["mean"].get(metric_name)
            std_val = agg["std"].get(metric_name)
            if mean_val is not None:
                lines.append(f"| {metric_name} | {mean_val:.4f} | {std_val:.4f} |")
            else:
                lines.append(f"| {metric_name} | N/A | N/A |")
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("![Class distribution](class_distribution.png)")
    lines.append("")
    for split_name in ["val", "test"]:
        lines.append(f"![Majority confusion matrix — {split_name}](majority_confusion_{split_name}.png)")
        lines.append("")
        lines.append(f"![Stratified boxplot — {split_name}](stratified_boxplot_{split_name}.png)")
        lines.append("")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run sanity-check baselines.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with train.csv, val.csv, test.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds for the stratified baseline")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (stratified seeds = seed..seed+n_seeds-1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SANITY-CHECK BASELINES")
    print("=" * 60)
    print(f"Data dir  : {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seeds     : {args.seed} .. {args.seed + args.n_seeds - 1}")
    print()

    # ── Load data ──
    splits = load_splits(args.data_dir)
    train_labels = splits["train"]["label"].values

    # Training class distribution
    train_dist = dict(splits["train"]["label"].value_counts().sort_index())
    total = sum(train_dist.values())
    print(f"\nTraining class distribution:")
    for label_val, count in sorted(train_dist.items()):
        print(f"  Label {label_val}: {count} ({count/total:.3f})")
    print()

    # ── Majority baseline ──
    print("-" * 40)
    print("MAJORITY BASELINE")
    print("-" * 40)

    majority_results = {}
    for split_name in ["val", "test"]:
        y_true = splits[split_name]["label"].values
        y_pred, y_prob, maj_label, pos_rate = run_majority_baseline(train_labels, y_true)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        majority_results[split_name] = metrics

        print(f"\n[{split_name}] Majority label={maj_label}, pos_rate={pos_rate:.4f}")
        print(format_metrics_block(metrics, indent="  "))

    # ── Stratified random baseline ──
    print()
    print("-" * 40)
    print("STRATIFIED RANDOM BASELINE")
    print("-" * 40)

    stratified_seed_rows = []
    stratified_all_metrics = {s: [] for s in ["val", "test"]}

    seeds = list(range(args.seed, args.seed + args.n_seeds))
    for seed in seeds:
        for split_name in ["val", "test"]:
            y_true = splits[split_name]["label"].values
            y_pred, y_prob = run_stratified_baseline(train_labels, y_true, seed)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            stratified_all_metrics[split_name].append(metrics)

            row = {"seed": seed, "split": split_name}
            for k, v in metrics.items():
                if k != "confusion_matrix":
                    row[k] = v
            stratified_seed_rows.append(row)

    # Aggregate mean / std
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    stratified_agg = {}
    for split_name in ["val", "test"]:
        vals = {m: [] for m in metric_cols}
        for metrics in stratified_all_metrics[split_name]:
            for m in metric_cols:
                v = metrics.get(m)
                if v is not None:
                    vals[m].append(v)

        mean_dict, std_dict = {}, {}
        for m in metric_cols:
            if vals[m]:
                mean_dict[m] = float(np.mean(vals[m]))
                std_dict[m] = float(np.std(vals[m]))
            else:
                mean_dict[m] = None
                std_dict[m] = None

        stratified_agg[split_name] = {"mean": mean_dict, "std": std_dict}

        print(f"\n[{split_name}] Stratified random (n_seeds={args.n_seeds}):")
        for m in metric_cols:
            if mean_dict[m] is not None:
                print(f"  {m}: {mean_dict[m]:.4f} ± {std_dict[m]:.4f}")
            else:
                print(f"  {m}: N/A")

    # ── Save outputs ──
    print()
    print("=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # 1. metrics.json
    json_out = {
        "majority": {s: majority_results[s] for s in ["val", "test"]},
        "stratified_random": {
            "aggregated": {s: stratified_agg[s] for s in ["val", "test"]},
        },
        "config": {
            "data_dir": args.data_dir,
            "n_seeds": args.n_seeds,
            "base_seed": args.seed,
            "seeds": seeds,
        },
    }
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Wrote {json_path}")

    # 2. summary.csv
    summary_rows = []
    for split_name in ["val", "test"]:
        row = {"split": split_name, "baseline": "majority"}
        for k, v in majority_results[split_name].items():
            if k != "confusion_matrix":
                row[k] = v
        summary_rows.append(row)

        row = {"split": split_name, "baseline": "stratified_random_mean"}
        for m in metric_cols:
            row[m] = stratified_agg[split_name]["mean"][m]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}")

    # 3. stratified_per_seed.csv
    seed_df = pd.DataFrame(stratified_seed_rows)
    seed_path = os.path.join(args.output_dir, "stratified_per_seed.csv")
    seed_df.to_csv(seed_path, index=False)
    print(f"  Wrote {seed_path}")

    # 4. report.md
    report_md = build_markdown_report(splits, train_dist, majority_results, stratified_agg, args.output_dir)
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  Wrote {report_path}")

    # 5. Plots
    plot_class_distribution(splits, os.path.join(args.output_dir, "class_distribution.png"))
    print(f"  Wrote class_distribution.png")

    for split_name in ["val", "test"]:
        cm = majority_results[split_name]["confusion_matrix"]
        plot_confusion_matrix(
            cm,
            f"Majority Baseline — {split_name}",
            os.path.join(args.output_dir, f"majority_confusion_{split_name}.png"),
        )
        print(f"  Wrote majority_confusion_{split_name}.png")

    for split_name in ["val", "test"]:
        split_seed_df = seed_df[seed_df["split"] == split_name]
        available_cols = [c for c in metric_cols if c in split_seed_df.columns and split_seed_df[c].notna().any()]
        if available_cols:
            plot_metric_boxplot(
                split_seed_df,
                available_cols,
                f"Stratified Random — {split_name} (n={args.n_seeds} seeds)",
                os.path.join(args.output_dir, f"stratified_boxplot_{split_name}.png"),
            )
            print(f"  Wrote stratified_boxplot_{split_name}.png")

    print()
    print("Done! See report at:", report_path)


if __name__ == "__main__":
    main()
