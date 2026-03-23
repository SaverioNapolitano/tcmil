#!/usr/bin/env python3
"""Hyperparameter sweep for the Dialogue Mean classifier head (frozen encoder).

Sweeps over:
    - lr:           1e-5, 3e-5, 1e-4, 3e-4
    - weight_decay: 0, 1e-4, 1e-2
    - dropout:      0.1, 0.3, 0.5

Embeddings are pre-computed once with frozen RoBERTa, then 36 classifier
heads are trained cheaply. Results are saved to outputs/dialogue_mean_sweep/.

Usage:
    cd /path/to/damil-2
    uv run python scripts/sweep_dialogue_mean.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_interviews, print_split_stats
from models.dialogue_mean import (
    DialogueMeanClassifier,
    encode_utterances,
    mean_pool_interview,
)
from utils.metrics import compute_metrics

# ──────────────────────────────────────────────────────────────────────
# Fixed configuration
# ──────────────────────────────────────────────────────────────────────
ENCODER_NAME = "roberta-base"
MAX_TOKEN_LENGTH = 128
ENCODING_BATCH_SIZE = 32
HIDDEN_DIM = 768
NUM_EPOCHS = 100
PATIENCE = 10
TRAIN_BATCH_SIZE = 16
SEED = 42
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dialogue_mean_sweep"

# ──────────────────────────────────────────────────────────────────────
# Sweep grid
# ──────────────────────────────────────────────────────────────────────
LR_VALUES = [1e-5, 3e-5, 1e-4, 3e-4]
WEIGHT_DECAY_VALUES = [0, 1e-4, 1e-2]
DROPOUT_VALUES = [0.1, 0.3, 0.5]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precompute_embeddings(
    interviews: list[dict],
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    device: torch.device,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pre-compute mean-pooled interview embeddings."""
    embeddings, labels, ids = [], [], []

    for iv in interviews:
        pid = iv["interview_id"]
        if len(iv["utterances"]) == 0:
            print(f"  [WARN] Interview {pid} has 0 utterances, using zero vector.")
            emb = torch.zeros(HIDDEN_DIM)
        else:
            utt_embs = encode_utterances(
                iv["utterances"], tokenizer, encoder, device,
                max_length=MAX_TOKEN_LENGTH, batch_size=ENCODING_BATCH_SIZE,
            )
            emb = mean_pool_interview(utt_embs)

        embeddings.append(emb)
        labels.append(iv["label"])
        ids.append(pid)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.float32)
    print(f"  {split_name}: {len(ids)} interviews encoded")
    return embeddings, labels, ids


def train_and_evaluate(
    lr: float,
    weight_decay: float,
    dropout: float,
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> dict:
    """Train a single classifier head configuration and return best val metrics.

    Returns:
        Dict with hyperparameters and best validation metrics.
    """
    set_seed(SEED)

    model = DialogueMeanClassifier(hidden_dim=HIDDEN_DIM, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_val_f1 = -1.0
    best_epoch = 0
    best_val_metrics = {}
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        n = len(train_labels)
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, TRAIN_BATCH_SIZE):
            batch_idx = indices[start : start + TRAIN_BATCH_SIZE]
            x = train_emb[batch_idx].to(device)
            y = train_labels[batch_idx].to(device)

            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # ── Evaluate on val ──
        model.eval()
        with torch.no_grad():
            val_logits = model(val_emb.to(device)).squeeze(-1)
            val_loss = criterion(val_logits, val_labels.to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)

        val_m = compute_metrics(val_labels.numpy(), val_preds, val_probs)

        # Early stopping on val F1
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_val_metrics = val_m.copy()
            best_val_metrics["loss"] = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        **{f"val_{k}": v for k, v in best_val_metrics.items()},
    }


def main() -> None:
    print("=" * 60)
    print("  Dialogue Mean — Hyperparameter Sweep")
    print("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n── Loading interviews ──")
    train_interviews = load_interviews(DATA_DIR, "train")
    val_interviews = load_interviews(DATA_DIR, "dev")

    print_split_stats(train_interviews, "train")
    print_split_stats(val_interviews, "dev")

    # ── Pre-compute embeddings (once) ──
    print(f"\n── Loading encoder: {ENCODER_NAME} ──")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME)
    encoder.eval().to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    print("\n── Pre-computing embeddings ──")
    train_emb, train_labels, _ = precompute_embeddings(
        train_interviews, tokenizer, encoder, device, "train"
    )
    val_emb, val_labels, _ = precompute_embeddings(
        val_interviews, tokenizer, encoder, device, "val"
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # pos_weight
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"\npos_weight={pos_weight.item():.3f}")

    # ── Sweep ──
    grid = list(itertools.product(LR_VALUES, WEIGHT_DECAY_VALUES, DROPOUT_VALUES))
    n_configs = len(grid)
    print(f"\n── Running sweep: {n_configs} configurations ──\n")

    results = []

    for i, (lr, wd, do) in enumerate(grid, 1):
        t0 = time.time()
        result = train_and_evaluate(
            lr=lr, weight_decay=wd, dropout=do,
            train_emb=train_emb, train_labels=train_labels,
            val_emb=val_emb, val_labels=val_labels,
            pos_weight=pos_weight, device=device,
        )
        elapsed = time.time() - t0
        results.append(result)

        print(
            f"  [{i:2d}/{n_configs}] "
            f"lr={lr:.0e} wd={wd:.0e} do={do:.1f} | "
            f"val_f1={result['val_f1']:.4f} "
            f"val_bacc={result['val_balanced_accuracy']:.4f} "
            f"val_roc={result['val_roc_auc']:.4f} | "
            f"epoch={result['best_epoch']:3d} ({elapsed:.1f}s)"
        )

    # ── Save results ──
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)

    # ── Print top 5 ──
    print("\n── Top 5 configurations by val F1 ──\n")
    top_cols = ["lr", "weight_decay", "dropout", "best_epoch",
                "val_f1", "val_balanced_accuracy", "val_roc_auc", "val_pr_auc"]
    print(results_df[top_cols].head(5).to_string(index=False))

    # ── Save summary ──
    best = results_df.iloc[0]
    summary = {
        "best_config": {
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
            "dropout": best["dropout"],
        },
        "best_val_metrics": {
            "f1": best["val_f1"],
            "balanced_accuracy": best["val_balanced_accuracy"],
            "roc_auc": best["val_roc_auc"],
            "pr_auc": best["val_pr_auc"],
            "accuracy": best["val_accuracy"],
            "precision": best["val_precision"],
            "recall": best["val_recall"],
        },
        "best_epoch": int(best["best_epoch"]),
        "n_configs_evaluated": n_configs,
        "encoder": ENCODER_NAME,
        "seed": SEED,
    }
    with open(OUTPUT_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Markdown report ──
    report_lines = [
        "# Dialogue Mean — Hyperparameter Sweep Report\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Encoder**: `{ENCODER_NAME}` (frozen)",
        f"**Configurations evaluated**: {n_configs}",
        f"**Seed**: {SEED}\n",
        "## Sweep Grid\n",
        f"- **lr**: {LR_VALUES}",
        f"- **weight_decay**: {WEIGHT_DECAY_VALUES}",
        f"- **dropout**: {DROPOUT_VALUES}\n",
        "## Best Configuration\n",
        f"- **lr**: {best['lr']:.0e}",
        f"- **weight_decay**: {best['weight_decay']:.0e}",
        f"- **dropout**: {best['dropout']:.1f}",
        f"- **Best epoch**: {int(best['best_epoch'])}\n",
        "## Best Validation Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for metric in ["f1", "balanced_accuracy", "roc_auc", "pr_auc",
                    "accuracy", "precision", "recall"]:
        report_lines.append(f"| {metric} | {best[f'val_{metric}']:.4f} |")

    report_lines.append("\n## All Results (sorted by val F1)\n")
    # Build markdown table manually (avoids tabulate dependency)
    report_lines.append("| " + " | ".join(top_cols) + " |")
    report_lines.append("| " + " | ".join(["---"] * len(top_cols)) + " |")
    for _, row in results_df[top_cols].iterrows():
        vals = []
        for col in top_cols:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        report_lines.append("| " + " | ".join(vals) + " |")

    report_text = "\n".join(report_lines)
    (OUTPUT_DIR / "sweep_report.md").write_text(report_text)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
