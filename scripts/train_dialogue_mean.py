#!/usr/bin/env python3
"""Train and evaluate the Dialogue Mean baseline with frozen encoder.

Usage:
    cd /path/to/damil-2
    uv run python scripts/train_dialogue_mean.py

All outputs are saved to outputs/dialogue_mean_frozen/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_interviews, print_split_stats
from models.dialogue_mean import (
    DialogueMeanClassifier,
    encode_utterances,
    mean_pool_interview,
)
from utils.metrics import compute_metrics, confusion_matrix_dict
from utils.plots import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_pr_curve,
    plot_probability_histogram,
    plot_roc_curve,
    plot_utterance_distribution,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
ENCODER_NAME = "roberta-base"     # Pretrained text encoder
POOLING = "cls"                   # CLS token pooling
MAX_TOKEN_LENGTH = 128            # Max tokens per utterance
ENCODING_BATCH_SIZE = 32          # Batch size for encoder

HIDDEN_DIM = 768                  # RoBERTa-base hidden size
DROPOUT = 0.1                     # Classifier dropout
LEARNING_RATE = 1e-3              # Adam LR for classifier head
WEIGHT_DECAY = 1e-4               # L2 regularization
NUM_EPOCHS = 100                  # Max epochs
PATIENCE = 10                     # Early stopping patience on val F1
TRAIN_BATCH_SIZE = 16             # Mini-batch size for classifier training

SEED = 42                         # Reproducibility
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dialogue_mean_frozen"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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
    embeddings = []
    labels = []
    ids = []
    norms = []

    for iv in interviews:
        utterances = iv["utterances"]
        label = iv["label"]
        pid = iv["interview_id"]

        if len(utterances) == 0:
            print(f"  [WARN] Interview {pid} has 0 utterances, using zero vector.")
            emb = torch.zeros(HIDDEN_DIM)
        else:
            utt_embs = encode_utterances(
                utterances, tokenizer, encoder, device,
                max_length=MAX_TOKEN_LENGTH, batch_size=ENCODING_BATCH_SIZE,
            )
            emb = mean_pool_interview(utt_embs)

        embeddings.append(emb)
        labels.append(label)
        ids.append(pid)
        norms.append(emb.norm().item())

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.float32)

    print(f"\n  {split_name} embedding stats:")
    print(f"    norm: mean={np.mean(norms):.3f}, std={np.std(norms):.3f}")

    return embeddings, labels, ids


def train_one_epoch(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the classifier for one epoch. Returns average loss."""
    model.train()
    n = len(labels)
    indices = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, TRAIN_BATCH_SIZE):
        batch_idx = indices[start : start + TRAIN_BATCH_SIZE]
        x = embeddings[batch_idx].to(device)
        y = labels[batch_idx].to(device)

        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model. Returns (loss, predictions, probabilities)."""
    model.eval()
    x = embeddings.to(device)
    y = labels.to(device)

    logits = model(x).squeeze(-1)
    loss = criterion(logits, y).item()

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    return loss, preds, probs


def generate_report(
    config: dict,
    metrics_all: dict,
    train_stats: dict,
    output_dir: Path,
) -> None:
    """Generate a human-readable markdown report."""
    report = []
    report.append("# Dialogue Mean Baseline — Experiment Report (Frozen Encoder)\n")
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Configuration\n")
    report.append(f"- **Encoder**: `{config['encoder']}` (frozen)")
    report.append(f"- **Pooling**: `{config['pooling']}` (CLS token)")
    report.append(f"- **Max token length**: {config['max_token_length']}")
    report.append(f"- **Hidden dim**: {config['hidden_dim']}")
    report.append(f"- **Dropout**: {config['dropout']}")
    report.append(f"- **Learning rate**: {config['learning_rate']}")
    report.append(f"- **Epochs trained**: {config['epochs_trained']}")
    report.append(f"- **Best epoch**: {config['best_epoch']}")
    report.append(f"- **Seed**: {config['seed']}")
    report.append("")

    report.append("## Training Summary\n")
    report.append(f"- Train interviews: {train_stats['n_train']}")
    report.append(f"- Val interviews: {train_stats['n_val']}")
    report.append(f"- Test interviews: {train_stats['n_test']}")
    report.append(f"- Positive weight (pos_weight): {train_stats['pos_weight']:.3f}")
    report.append("")

    for split_name in ["val", "test"]:
        report.append(f"## {split_name.capitalize()} Results\n")
        m = metrics_all[split_name]
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for key, val in m.items():
            if key == "confusion_matrix":
                continue
            report.append(f"| {key} | {val:.4f} |")
        cm = m.get("confusion_matrix", {})
        if cm:
            report.append(f"\n**Confusion Matrix**: TP={cm['tp']}, FP={cm['fp']}, "
                          f"TN={cm['tn']}, FN={cm['fn']}")
        report.append("")

    report.append("## Plots\n")
    report.append(f"See the `outputs/{output_dir.name}/` directory for all plots.")
    report.append("")

    report.append("## Limitations\n")
    report.append("- No fine-tuning of the pretrained encoder (frozen features only).")
    report.append("- Speaker role information is ignored.")
    report.append("- No utterance weighting — all utterances contribute equally.")
    report.append("- Small dataset (~190 interviews total) limits generalization.")
    report.append("")

    report_text = "\n".join(report)
    (output_dir / "report.md").write_text(report_text)
    print("\n" + report_text)


def main() -> None:
    print("=" * 60)
    print("  Dialogue Mean Baseline — Training & Evaluation (Frozen)")
    print("=" * 60)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Encoder: {ENCODER_NAME} (frozen)")
    print(f"Pooling: {POOLING}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n── Loading interviews ──")
    train_interviews = load_interviews(DATA_DIR, "train")
    val_interviews = load_interviews(DATA_DIR, "dev")
    test_interviews = load_interviews(DATA_DIR, "test")

    utterance_counts = {}
    utterance_counts["train"] = print_split_stats(train_interviews, "train")
    utterance_counts["val"] = print_split_stats(val_interviews, "dev")
    utterance_counts["test"] = print_split_stats(test_interviews, "test")

    # ── Load encoder ──
    print(f"\n── Loading encoder: {ENCODER_NAME} ──")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME)
    encoder.eval()
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    # ── Pre-compute embeddings ──
    print("\n── Pre-computing interview embeddings ──")
    train_emb, train_labels, train_ids = precompute_embeddings(
        train_interviews, tokenizer, encoder, device, "train"
    )
    val_emb, val_labels, val_ids = precompute_embeddings(
        val_interviews, tokenizer, encoder, device, "val"
    )
    test_emb, test_labels, test_ids = precompute_embeddings(
        test_interviews, tokenizer, encoder, device, "test"
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute pos_weight ──
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"\npos_weight={pos_weight.item():.3f}")

    # ── Initialize model ──
    model = DialogueMeanClassifier(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print(f"\nClassifier: {model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Training loop ──
    print("\n── Training ──")
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_balanced_accuracy": [], "val_balanced_accuracy": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_emb, train_labels, optimizer, criterion, device
        )

        _, train_preds, train_probs = evaluate(
            model, train_emb, train_labels, criterion, device
        )
        val_loss, val_preds, val_probs = evaluate(
            model, val_emb, val_labels, criterion, device
        )

        train_m = compute_metrics(train_labels.numpy(), train_preds, train_probs)
        val_m = compute_metrics(val_labels.numpy(), val_preds, val_probs)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])
        history["train_balanced_accuracy"].append(train_m["balanced_accuracy"])
        history["val_balanced_accuracy"].append(val_m["balanced_accuracy"])

        marker = ""
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            marker = " ★"
        else:
            patience_counter += 1

        if epoch <= 3 or epoch % 5 == 0 or epoch == NUM_EPOCHS or patience_counter >= PATIENCE:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_f1={train_m['f1']:.4f} val_f1={val_m['f1']:.4f} | "
                f"val_bacc={val_m['balanced_accuracy']:.4f}{marker}"
            )

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE}).")
            break

    print(f"\n  Best epoch: {best_epoch} (val F1={best_val_f1:.4f})")

    # ── Evaluate best model ──
    print("\n── Evaluating best model ──")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", weights_only=True))

    metrics_all = {}
    predictions_rows = []

    for split_name, emb, labels, ids in [
        ("val", val_emb, val_labels, val_ids),
        ("test", test_emb, test_labels, test_ids),
    ]:
        loss, preds, probs = evaluate(model, emb, labels, criterion, device)
        y_true = labels.numpy()
        m = compute_metrics(y_true, preds, probs)
        cm = confusion_matrix_dict(y_true, preds)
        m["confusion_matrix"] = cm

        metrics_all[split_name] = m

        print(f"\n  {split_name.upper()} metrics:")
        for k, v in m.items():
            if k != "confusion_matrix":
                print(f"    {k}: {v:.4f}")
        print(f"    confusion_matrix: {cm}")

        for pid, yt, yp, prob in zip(ids, y_true, preds, probs):
            predictions_rows.append({
                "interview_id": pid,
                "split": split_name,
                "true_label": int(yt),
                "predicted_label": int(yp),
                "probability": float(prob),
            })

        plot_roc_curve(y_true, probs, split_name, OUTPUT_DIR)
        plot_pr_curve(y_true, probs, split_name, OUTPUT_DIR)
        plot_confusion_matrix(y_true, preds, split_name, OUTPUT_DIR)
        plot_probability_histogram(y_true, probs, split_name, OUTPUT_DIR)

    # ── Save outputs ──
    print("\n── Saving outputs ──")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print("  Saved metrics.json")

    pred_df = pd.DataFrame(predictions_rows)
    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"  Saved predictions.csv ({len(pred_df)} rows)")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / "train_history.csv", index=False)
    print(f"  Saved train_history.csv ({len(hist_df)} rows)")

    plot_loss_curves(history["train_loss"], history["val_loss"], OUTPUT_DIR)
    plot_metric_curves(history, ["f1", "balanced_accuracy"], OUTPUT_DIR)
    plot_utterance_distribution(utterance_counts, OUTPUT_DIR)
    print("  Saved all plots")

    config = {
        "encoder": ENCODER_NAME,
        "pooling": POOLING,
        "max_token_length": MAX_TOKEN_LENGTH,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "seed": SEED,
    }
    train_stats = {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "pos_weight": pos_weight.item(),
    }
    generate_report(config, metrics_all, train_stats, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
