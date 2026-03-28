"""Training script for DAMIL-H (Hierarchical Dual Attention MIL) baseline.

Pre-computes DistilBERT [CLS] embeddings for all utterances, then trains
a lightweight attention-based MIL classifier on the frozen representations.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews, print_split_stats
from models.damil_h import DAMILHClassifier
from utils.metrics import (
    compute_attention_entropy,
    compute_metrics,
    confusion_matrix_dict,
    find_best_threshold,
)
from utils.plots import (
    plot_attention_entropy,
    plot_attention_weights_bar,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_pr_curve,
    plot_prob_vs_bag_size,
    plot_probability_histogram,
    plot_roc_curve,
    plot_utterance_distribution,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Embedding extraction (frozen encoder)
# ---------------------------------------------------------------------------

ENCODER_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 64
ENCODING_BATCH_SIZE = 64


@torch.no_grad()
def encode_utterances(
    utterances: list[str],
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int = MAX_TOKEN_LENGTH,
    batch_size: int = ENCODING_BATCH_SIZE,
) -> torch.Tensor:
    """Encode a list of utterances into [CLS] embeddings with a frozen encoder.

    Args:
        utterances: Raw text strings.
        tokenizer: HuggingFace tokenizer.
        encoder: Pretrained transformer model (frozen).
        device: Torch device.
        max_length: Maximum token length per utterance.
        batch_size: Encoding batch size.

    Returns:
        Tensor of shape (num_utterances, hidden_size).
    """
    encoder.eval()
    all_embeddings = []

    for start in range(0, len(utterances), batch_size):
        batch_texts = utterances[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def precompute_embeddings(
    interviews: list[dict],
    tokenizer,
    encoder,
    device: torch.device,
) -> list[dict]:
    """Pre-compute embeddings for all interviews.

    Args:
        interviews: List of interview dicts with 'utterances' key.
        tokenizer: HuggingFace tokenizer.
        encoder: Frozen pretrained encoder.
        device: Torch device.

    Returns:
        List of dicts with added 'embeddings' key (Tensor of shape [N, hidden_size]).
    """
    enriched = []
    for iv in interviews:
        utts = iv["utterances"] if iv["utterances"] else [""]
        emb = encode_utterances(utts, tokenizer, encoder, device)
        enriched.append({**iv, "embeddings": emb})
    return enriched


# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------

class EmbeddedBagDataset(Dataset):
    """Dataset wrapping pre-computed utterance embeddings per interview."""

    def __init__(self, interviews: list[dict]):
        self.interviews = interviews

    def __len__(self) -> int:
        return len(self.interviews)

    def __getitem__(self, idx: int) -> dict:
        return self.interviews[idx]


def collate_embedded_bags(batch: list[dict]) -> dict:
    """Collate pre-computed embeddings into a padded batch.

    Returns:
        Dictionary with keys:
            interview_ids: list[int]
            labels: Tensor (batch_size,)
            bags: Tensor (batch_size, max_turns, embedding_dim) — zero-padded
            bag_sizes: list[int]
            utterances_lists: list[list[str]]
    """
    interview_ids = []
    labels = []
    embeddings_list = []
    bag_sizes = []
    utterances_lists = []

    for item in batch:
        interview_ids.append(item["interview_id"])
        labels.append(item["label"])
        embeddings_list.append(item["embeddings"])
        bag_sizes.append(item["embeddings"].size(0))
        utts = item["utterances"] if item["utterances"] else [""]
        utterances_lists.append(utts)

    # Pad to max bag size in this batch
    max_turns = max(bag_sizes)
    embedding_dim = embeddings_list[0].size(1)
    padded = torch.zeros(len(batch), max_turns, embedding_dim)
    for i, emb in enumerate(embeddings_list):
        padded[i, : emb.size(0), :] = emb

    return {
        "interview_ids": interview_ids,
        "labels": torch.tensor(labels, dtype=torch.float),
        "bags": padded,
        "bag_sizes": bag_sizes,
        "utterances_lists": utterances_lists,
    }


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: DAMILHClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    entropy_lambda: float = 0.0,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        (average_loss, f1_score)
    """
    from sklearn.metrics import f1_score

    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        bags = batch["bags"].to(device)
        labels = batch["labels"].to(device)
        bag_sizes = batch["bag_sizes"]

        optimizer.zero_grad()
        logits, att_weights_list = model.forward_batch(bags, bag_sizes)
        loss = criterion(logits, labels)

        if entropy_lambda > 0.0:
            entropies = []
            for aw in att_weights_list:
                entropies.append(-torch.sum(aw * torch.log(aw + 1e-9)))
            mean_entropy = torch.stack(entropies).mean()
            loss = loss - entropy_lambda * mean_entropy

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(
    model: DAMILHClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict, dict]:
    """Evaluate the model and return loss, metrics, and detailed predictions.

    Returns:
        (avg_loss, metrics_dict, predictions_dict)
    """
    model.eval()
    total_loss = 0.0

    all_ids = []
    all_labels = []
    all_probs = []
    all_preds = []
    all_bag_sizes = []
    all_attention_weights = []
    all_entropies = []
    all_utterance_texts = []

    for batch in dataloader:
        bags = batch["bags"].to(device)
        labels = batch["labels"].to(device)
        bag_sizes = batch["bag_sizes"]
        utterances_lists = batch["utterances_lists"]

        logits, att_weights_list = model.forward_batch(bags, bag_sizes)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_ids.extend(batch["interview_ids"])
        all_labels.extend(batch["labels"].numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_bag_sizes.extend(bag_sizes)
        all_utterance_texts.extend(utterances_lists)

        for aw in att_weights_list:
            aw_np = aw.cpu().numpy()
            all_attention_weights.append(aw_np)
            all_entropies.append(compute_attention_entropy(aw_np))

    avg_loss = total_loss / len(dataloader)

    metrics = compute_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
    )

    predictions = {
        "interview_id": all_ids,
        "true_label": [int(l) for l in all_labels],
        "predicted_label": all_preds,
        "probability": all_probs,
        "num_utterances": all_bag_sizes,
        "attention_weights": all_attention_weights,
        "attention_entropy": all_entropies,
        "utterance_texts": all_utterance_texts,
    }

    return avg_loss, metrics, predictions


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DAMIL-H (Hierarchical Dual Attention MIL)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_h")
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_TOKEN_LENGTH)
    parser.add_argument("--proj_dim", type=int, default=0, help="Projection dim (0 to disable)")
    parser.add_argument("--att_hidden_dim", type=int, default=32)
    parser.add_argument("--attention_temp", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--entropy_lambda", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(out_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading interviews...")
    train_data = load_interviews(args.data_dir, "train")
    dev_data = load_interviews(args.data_dir, "dev")

    print_split_stats(train_data, "train")
    print_split_stats(dev_data, "dev")

    # --- Pre-compute embeddings ---
    logger.info(f"Pre-computing embeddings with frozen {args.encoder_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_data = precompute_embeddings(train_data, tokenizer, encoder, device)
    dev_data = precompute_embeddings(dev_data, tokenizer, encoder, device)

    embedding_dim = train_data[0]["embeddings"].size(1)
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Dataloaders ---
    train_loader = DataLoader(
        EmbeddedBagDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_embedded_bags,
    )
    dev_loader = DataLoader(
        EmbeddedBagDataset(dev_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_embedded_bags,
    )

    # --- Model Setup ---
    proj_dim = args.proj_dim if args.proj_dim > 0 else None
    model = DAMILHClassifier(
        embedding_dim=embedding_dim,
        proj_dim=proj_dim,
        att_hidden_dim=args.att_hidden_dim,
        dropout_rate=args.dropout_rate,
        temperature=args.attention_temp,
    )
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    num_pos = sum(1 for iv in train_data if iv["label"] == 1)
    num_neg = len(train_data) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Class balance: {num_pos} pos / {num_neg} neg — pos_weight={pos_weight.item():.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_balanced_accuracy": [],
    }

    logger.info("Starting training...")
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device,
            entropy_lambda=args.entropy_lambda,
        )
        val_loss, val_metrics, _ = evaluate(model, dev_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val BAcc: {val_metrics['balanced_accuracy']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            logger.info("  -> Found new best model (lowest val loss): saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(
                    f"Early stopping triggered after {epochs_no_improve} epochs "
                    f"without improvement."
                )
                break

    # Save training history
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    plot_loss_curves(history["train_loss"], history["val_loss"], out_dir)
    plot_metric_curves(history, ["f1", "balanced_accuracy"], out_dir)

    # --- Final Evaluation ---
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))

    # Tune threshold on dev
    logger.info("Evaluating on DEV split to tune threshold based on weighted loss...")
    _, _, dev_preds_default = evaluate(model, dev_loader, criterion, device, threshold=0.5)
    best_t = find_best_threshold(
        y_true=np.array(dev_preds_default["true_label"]),
        y_prob=np.array(dev_preds_default["probability"]),
        metric="loss",
        pos_weight=pos_weight.item(),
    )
    logger.info(f"Best tuned threshold on DEV (min weighted loss): {best_t:.4f}")

    # Re-evaluate with tuned threshold
    _, dev_metrics, dev_preds = evaluate(model, dev_loader, criterion, device, threshold=best_t)

    # Load and evaluate test set
    test_data = load_interviews(args.data_dir, "test")
    print_split_stats(test_data, "test")
    test_data = precompute_embeddings(
        test_data, tokenizer,
        AutoModel.from_pretrained(args.encoder_name).to(device).eval(),
        device,
    )
    test_loader = DataLoader(
        EmbeddedBagDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_embedded_bags,
    )

    logger.info("Evaluating on TEST split...")
    _, test_metrics, test_preds = evaluate(model, test_loader, criterion, device, threshold=best_t)

    # --- Save metrics ---
    final_metrics = {"dev": dev_metrics, "test": test_metrics, "threshold": best_t}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # --- Save predictions ---
    dev_df = pd.DataFrame(
        {k: v for k, v in dev_preds.items() if k not in ["attention_weights", "utterance_texts"]}
    )
    dev_df.insert(1, "split", "dev")
    test_df = pd.DataFrame(
        {k: v for k, v in test_preds.items() if k not in ["attention_weights", "utterance_texts"]}
    )
    test_df.insert(1, "split", "test")

    all_preds_df = pd.concat([dev_df, test_df], ignore_index=True)
    all_preds_df.to_csv(out_dir / "predictions.csv", index=False)

    # --- Generate plots ---
    logger.info("Generating evaluation plots and attention reports...")
    all_attention_records = []

    for split_name, df, preds_dict in [
        ("dev", dev_df, dev_preds),
        ("test", test_df, test_preds),
    ]:
        y_true = df["true_label"].values
        y_pred = df["predicted_label"].values
        y_prob = df["probability"].values
        bag_sizes = df["num_utterances"].values

        plot_roc_curve(y_true, y_prob, split_name, out_dir)
        plot_pr_curve(y_true, y_prob, split_name, out_dir)
        plot_confusion_matrix(y_true, y_pred, split_name, out_dir)
        plot_probability_histogram(y_true, y_prob, split_name, out_dir)
        plot_prob_vs_bag_size(bag_sizes, y_prob, split_name, out_dir)
        plot_attention_entropy(preds_dict["attention_entropy"], split_name, out_dir)

        # Collect attention weights for JSONL
        for i, iv_id in enumerate(preds_dict["interview_id"]):
            weights = preds_dict["attention_weights"][i]
            texts = preds_dict["utterance_texts"][i]
            for u_idx, (w, t) in enumerate(zip(weights, texts)):
                all_attention_records.append({
                    "interview_id": iv_id,
                    "split": split_name,
                    "utterance_index": u_idx,
                    "attention_weight": float(w),
                    "utterance_text": t,
                })

    # Utterance distribution
    utterance_counts = {
        "train": [len(iv["utterances"]) for iv in train_data],
        "dev": dev_df["num_utterances"].tolist(),
        "test": test_df["num_utterances"].tolist(),
    }
    plot_utterance_distribution(utterance_counts, out_dir)

    # Save attention weights JSONL
    with open(out_dir / "attention_weights.jsonl", "w") as f:
        for r in all_attention_records:
            f.write(json.dumps(r) + "\n")

    # --- Qualitative attention report ---
    attention_examples_md = "# DAMIL-H Attention Interpretation Examples\n\n"
    num_test_samples = min(5, len(test_preds["interview_id"]))
    if num_test_samples > 0:
        sample_indices = np.random.choice(
            len(test_preds["interview_id"]), num_test_samples, replace=False
        )
        for idx in sample_indices:
            iv_id = test_preds["interview_id"][idx]
            weights = test_preds["attention_weights"][idx]
            texts = test_preds["utterance_texts"][idx]
            true_label = test_preds["true_label"][idx]
            prob = test_preds["probability"][idx]

            example_dict = {
                "interview_id": iv_id,
                "true_label": true_label,
                "probability": prob,
                "attention_weights": weights,
                "utterance_texts": texts,
            }
            plot_attention_weights_bar(example_dict, out_dir, f"attention_bar_{iv_id}.png")

            attention_examples_md += f"## Interview: {iv_id}\n"
            attention_examples_md += f"- **True Label**: {true_label}\n"
            attention_examples_md += f"- **Predicted Probability**: {prob:.4f}\n\n"
            attention_examples_md += "### Top 5 Attended Utterances\n"

            top_indices = np.argsort(weights)[::-1][: min(5, len(weights))]
            for t_idx in top_indices:
                attention_examples_md += (
                    f"**[{t_idx}]** (weight: {weights[t_idx]:.4f}): {texts[t_idx]}\n\n"
                )

        with open(out_dir / "attention_examples.md", "w") as f:
            f.write(attention_examples_md)

    # --- Summary ---
    with open(out_dir / "summary.md", "w") as f:
        f.write("# DAMIL-H (Hierarchical Dual Attention MIL) Baseline\n\n")
        f.write(f"Encoder: `{args.encoder_name}` (frozen)\n\n")
        f.write("## Metrics\n```json\n")
        f.write(json.dumps(final_metrics, indent=4))
        f.write("\n```\n\n")
        f.write("## Sample Predictions\n")
        sample_df = all_preds_df.sample(min(15, len(all_preds_df)), random_state=args.seed)
        cols = ["interview_id", "split", "num_utterances", "true_label", "predicted_label", "probability"]
        f.write(sample_df[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    logger.info("Done!")


if __name__ == "__main__":
    main()
