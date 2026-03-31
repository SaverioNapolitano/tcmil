"""Training script for DAMIL-R (Role-Aware Dual Attention MIL).

Pre-computes DistilBERT [CLS] embeddings for both participant and interviewer
utterances, then trains a cross-attention MIL classifier on frozen representations.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews_with_roles, print_split_stats
from models.damil_r import DAMILRClassifier
from utils.metrics import (
    compute_attention_entropy,
    compute_metrics,
    find_best_threshold,
)
from utils.plots import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_pr_curve,
    plot_roc_curve,
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
# Constants
# ---------------------------------------------------------------------------

ENCODER_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 128


# ---------------------------------------------------------------------------
# Datasets and Collation
# ---------------------------------------------------------------------------

class DualRoleBagDataset(Dataset):
    """Dataset for pre-computed utterance embeddings with both roles."""

    def __init__(self, interviews: list[dict]):
        self.interviews = interviews

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        item = self.interviews[idx]
        return {
            "patient_bag": item["patient_embeddings"],      # (P, d)
            "interviewer_bag": item["interviewer_embeddings"],  # (I, d)
            "label": torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
            "utterances": item.get("utterances", []),
            "interviewer_utterances": item.get("interviewer_utterances", []),
        }


def collate_dual_role_bags(batch):
    """Collate dual-role bags with independent padding for each role."""
    patient_bags = [item["patient_bag"] for item in batch]
    interviewer_bags = [item["interviewer_bag"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["interview_id"] for item in batch]
    utts = [item["utterances"] for item in batch]
    int_utts = [item["interviewer_utterances"] for item in batch]

    patient_sizes = [bag.size(0) for bag in patient_bags]
    interviewer_sizes = [bag.size(0) for bag in interviewer_bags]

    max_p = max(patient_sizes)
    max_i = max(interviewer_sizes)
    d = patient_bags[0].size(1)

    padded_patient = torch.zeros(len(batch), max_p, d)
    padded_interviewer = torch.zeros(len(batch), max_i, d)

    for idx, (p_bag, i_bag) in enumerate(zip(patient_bags, interviewer_bags)):
        padded_patient[idx, :patient_sizes[idx], :] = p_bag
        padded_interviewer[idx, :interviewer_sizes[idx], :] = i_bag

    return {
        "patient_bags": padded_patient,
        "interviewer_bags": padded_interviewer,
        "patient_sizes": patient_sizes,
        "interviewer_sizes": interviewer_sizes,
        "labels": labels,
        "interview_ids": ids,
        "utterances_lists": utts,
        "interviewer_utterances_lists": int_utts,
    }


# ---------------------------------------------------------------------------
# Embedding Pre-computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_dual_role_embeddings(
    interviews: list[dict],
    tokenizer,
    encoder,
    device,
    max_len: int = MAX_TOKEN_LENGTH,
) -> list[dict]:
    """Pre-compute DistilBERT [CLS] embeddings for both participant and interviewer utterances.

    Args:
        interviews: List of interview dicts with 'utterances' and 'interviewer_utterances'.
        tokenizer: HuggingFace tokenizer.
        encoder: HuggingFace transformer model.
        device: Torch device.
        max_len: Maximum token length for truncation.

    Returns:
        Enriched interview list with 'patient_embeddings' and 'interviewer_embeddings' tensors.
    """
    encoder.eval()
    processed = []

    for iv in tqdm(interviews, desc="Pre-computing dual-role embeddings"):
        # Patient utterances
        patient_utts = iv.get("utterances", [""])
        if not patient_utts:
            patient_utts = [""]

        encoded_p = tokenizer(
            patient_utts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        patient_emb = encoder(**encoded_p).last_hidden_state[:, 0, :].cpu()

        # Interviewer utterances
        interviewer_utts = iv.get("interviewer_utterances", [""])
        if not interviewer_utts:
            interviewer_utts = [""]

        encoded_i = tokenizer(
            interviewer_utts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        interviewer_emb = encoder(**encoded_i).last_hidden_state[:, 0, :].cpu()

        processed.append({
            **iv,
            "patient_embeddings": patient_emb,
            "interviewer_embeddings": interviewer_emb,
        })

    return processed


# ---------------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, criterion, optimizer, device,
    entropy_lambda=0.0, max_grad_norm=1.0,
):
    """Train for one epoch.

    Args:
        model: DAMILRClassifier.
        loader: DataLoader yielding dual-role batches.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: Optimizer.
        device: Torch device.
        entropy_lambda: Coefficient for turn-attention entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        avg_loss: Average training loss.
        metrics: Dictionary of training metrics.
    """
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        optimizer.zero_grad()
        target = batch["labels"].to(device)

        patient_bags = batch["patient_bags"].to(device)
        interviewer_bags = batch["interviewer_bags"].to(device)
        patient_sizes = batch["patient_sizes"]
        interviewer_sizes = batch["interviewer_sizes"]

        logits, cross_attn_list, turn_attn_list = model.forward_batch(
            patient_bags, interviewer_bags, patient_sizes, interviewer_sizes,
        )

        loss = criterion(logits, target)

        # Optional entropy regularization on turn-level attention
        if entropy_lambda > 0:
            entropy = 0
            for turn_w in turn_attn_list:
                entropy -= torch.sum(turn_w * torch.log(turn_w + 1e-9))
            loss = loss - entropy_lambda * (entropy / len(turn_attn_list))

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_labels),
        (np.array(all_probs) >= 0.5).astype(int),
        np.array(all_probs),
    )

    return avg_loss, metrics


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model on a data loader.

    Args:
        model: DAMILRClassifier.
        loader: DataLoader yielding dual-role batches.
        criterion: Loss function.
        device: Torch device.
        threshold: Classification threshold.

    Returns:
        avg_loss: Average evaluation loss.
        metrics: Dictionary of evaluation metrics.
        predictions: Dictionary with per-sample results including attention weights.
    """
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_ids = []
    all_cross_attn = []
    all_turn_attn = []
    all_utterances = []
    all_int_utterances = []
    all_cross_entropies = []
    all_turn_entropies = []

    with torch.no_grad():
        for batch in loader:
            target = batch["labels"].to(device)

            patient_bags = batch["patient_bags"].to(device)
            interviewer_bags = batch["interviewer_bags"].to(device)
            patient_sizes = batch["patient_sizes"]
            interviewer_sizes = batch["interviewer_sizes"]

            logits, cross_attn_list, turn_attn_list = model.forward_batch(
                patient_bags, interviewer_bags, patient_sizes, interviewer_sizes,
            )

            loss = criterion(logits, target)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

            for ca in cross_attn_list:
                ca_np = ca.cpu().numpy()
                all_cross_attn.append(ca_np)
                # Mean entropy across patient turns' attention over interviewer
                row_entropies = []
                for row in ca_np:
                    row_entropies.append(compute_attention_entropy(row))
                all_cross_entropies.append(float(np.mean(row_entropies)))

            for ta in turn_attn_list:
                ta_np = ta.cpu().numpy()
                all_turn_attn.append(ta_np)
                all_turn_entropies.append(compute_attention_entropy(ta_np))

            if "utterances_lists" in batch:
                all_utterances.extend(batch["utterances_lists"])
            if "interviewer_utterances_lists" in batch:
                all_int_utterances.extend(batch["interviewer_utterances_lists"])

    avg_loss = total_loss / len(loader)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    predictions = {
        "interview_id": all_ids,
        "true_label": [int(l) for l in all_labels],
        "probability": [float(p) for p in all_probs],
        "predicted_label": [int(p) for p in y_pred],
        "num_utterances": [len(u) for u in all_utterances] if all_utterances else [],
        "cross_attention_weights": all_cross_attn,
        "turn_attention_weights": all_turn_attn,
        "cross_attention_entropy": all_cross_entropies,
        "turn_attention_entropy": all_turn_entropies,
        "utterance_texts": all_utterances if all_utterances else None,
        "interviewer_utterance_texts": all_int_utterances if all_int_utterances else None,
    }

    return avg_loss, metrics, predictions


def generate_attention_report(
    test_results: dict, output_path: Path, n_top: int = 5, n_examples: int = 10,
):
    """Generate a markdown report of top cross-attended interviewer turns per patient turn."""
    with open(output_path, "w") as f:
        f.write("# DAMIL-R Cross-Attention Behavior Report\n\n")
        f.write(
            "This report shows the top-attended interviewer utterances for each "
            "patient turn, for the first few test examples.\n\n"
        )

        for i in range(min(n_examples, len(test_results["interview_id"]))):
            sample_id = test_results["interview_id"][i]
            label = "Depressed" if test_results["true_label"][i] == 1 else "Not Depressed"
            prob = test_results["probability"][i]
            cross_attn = test_results["cross_attention_weights"][i]  # (P, I)
            turn_attn = test_results["turn_attention_weights"][i]    # (P,)
            p_utts = test_results["utterance_texts"][i] if test_results["utterance_texts"] else None
            i_utts = test_results["interviewer_utterance_texts"][i] if test_results["interviewer_utterance_texts"] else None

            f.write(f"## Interview {sample_id} ({label}, Pred Prob: {prob:.4f})\n\n")

            if p_utts is None or i_utts is None:
                f.write("Utterance texts not available.\n\n")
                continue

            # Top patient turns by turn-level attention
            top_p_indices = np.argsort(turn_attn)[::-1][:n_top]

            f.write(f"### Top-{n_top} Patient Turns (by turn-level attention)\n\n")
            for rank, p_idx in enumerate(top_p_indices, 1):
                f.write(f"**P{p_idx}** (turn_attn={turn_attn[p_idx]:.4f}): {p_utts[p_idx]}\n\n")

                # Top interviewer turns this patient turn attends to
                cross_row = cross_attn[p_idx]
                top_i_indices = np.argsort(cross_row)[::-1][:3]
                for i_idx in top_i_indices:
                    if i_idx < len(i_utts):
                        f.write(f"  - I{i_idx} (cross_attn={cross_row[i_idx]:.4f}): {i_utts[i_idx]}\n")
                f.write("\n")

            f.write("---\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DAMIL-R for Depression Detection")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_r")

    # Model Config
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_TOKEN_LENGTH)
    parser.add_argument("--proj_dim", type=int, default=0,
                        help="Projection dim before cross-attention. 0 = no projection.")
    parser.add_argument("--att_hidden_dim", type=int, default=64)
    parser.add_argument("--attention_temp", type=float, default=1.0)

    # Training Config
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss",
                        choices=["val_loss", "f1", "roc_auc", "pr_auc", "balanced_accuracy"])

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "train.log"), logging.StreamHandler()],
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
    logger.info("Loading interviews with both roles...")
    train_ivs = load_interviews_with_roles(args.data_dir, split="train")
    dev_ivs = load_interviews_with_roles(args.data_dir, split="dev")
    test_ivs = load_interviews_with_roles(args.data_dir, split="test")

    logger.info(f"Train: {len(train_ivs)}, Dev: {len(dev_ivs)}, Test: {len(test_ivs)}")

    # Pre-compute embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embedding_dim = encoder.config.hidden_size

    logger.info("Pre-computing embeddings for both roles...")
    train_ivs = precompute_dual_role_embeddings(train_ivs, tokenizer, encoder, device, args.max_len)
    dev_ivs = precompute_dual_role_embeddings(dev_ivs, tokenizer, encoder, device, args.max_len)
    test_ivs = precompute_dual_role_embeddings(test_ivs, tokenizer, encoder, device, args.max_len)

    train_dataset = DualRoleBagDataset(train_ivs)
    dev_dataset = DualRoleBagDataset(dev_ivs)
    test_dataset = DualRoleBagDataset(test_ivs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

    # --- Model ---
    proj_dim = args.proj_dim if args.proj_dim > 0 else None
    model = DAMILRClassifier(
        embedding_dim=embedding_dim,
        proj_dim=proj_dim,
        att_hidden_dim=args.att_hidden_dim,
        dropout_rate=args.dropout_rate,
        temperature=args.attention_temp,
    ).to(device)

    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class weighting
    num_pos = sum(1 for iv in train_ivs if iv["label"] == 1)
    num_neg = len(train_ivs) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Class weighting: pos_weight={pos_weight.item():.4f} (pos={num_pos}, neg={num_neg})")

    # --- Training Loop ---
    logger.info("Starting training...")
    history = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_bacc": [], "val_roc_auc": [], "val_pr_auc": [],
    }
    best_score = -float("inf") if args.checkpoint_metric != "val_loss" else float("inf")
    epochs_no_improve = 0
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, tr_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            entropy_lambda=args.entropy_lambda, max_grad_norm=args.max_grad_norm,
        )
        v_loss, v_metrics, _ = evaluate(model, dev_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_metrics["f1"])
        history["val_bacc"].append(v_metrics["balanced_accuracy"])
        history["val_roc_auc"].append(v_metrics["roc_auc"])
        history["val_pr_auc"].append(v_metrics["pr_auc"])

        logger.info(
            f"Epoch {epoch:02d} | Loss: {tr_loss:.4f}/{v_loss:.4f} | "
            f"F1: {v_metrics['f1']:.4f} | PR-AUC: {v_metrics['pr_auc']:.4f} | "
            f"ROC-AUC: {v_metrics['roc_auc']:.4f}"
        )

        score = v_metrics[args.checkpoint_metric] if args.checkpoint_metric != "val_loss" else v_loss
        is_best = (score > best_score) if args.checkpoint_metric != "val_loss" else (score < best_score)

        if is_best:
            best_score = score
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> Best model saved ({args.checkpoint_metric}: {score:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info("Early stopping.")
                break

    # --- Final Evaluation ---
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Tune threshold on dev
    _, _, v_preds = evaluate(model, dev_loader, criterion, device, threshold=0.5)
    best_t = find_best_threshold(
        np.array(v_preds["true_label"]),
        np.array(v_preds["probability"]),
        metric="loss",
        pos_weight=pos_weight.item(),
    )
    logger.info(f"Best threshold (tuned on dev): {best_t:.4f}")

    # Test evaluation
    _, test_metrics, test_results = evaluate(model, test_loader, criterion, device, threshold=best_t)
    logger.info(
        f"TEST | F1={test_metrics['f1']:.4f} | BAcc={test_metrics['balanced_accuracy']:.4f} | "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} | PR-AUC={test_metrics['pr_auc']:.4f}"
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": test_metrics, "threshold": best_t}, f, indent=4)

    # --- Plotting ---
    logger.info("Generating evaluation plots...")
    plot_loss_curves(history["train_loss"], history["val_loss"], out_dir)
    plot_metric_curves(history, ["f1", "balanced_accuracy"], out_dir)
    plot_roc_curve(
        np.array(test_results["true_label"]),
        np.array(test_results["probability"]),
        "test", out_dir,
    )
    plot_pr_curve(
        np.array(test_results["true_label"]),
        np.array(test_results["probability"]),
        "test", out_dir,
    )
    plot_confusion_matrix(
        np.array(test_results["true_label"]),
        np.array(test_results["predicted_label"]),
        "test", out_dir,
    )

    # Cross-attention heatmaps for a few samples
    from plots.cross_attention import (
        plot_cross_attention_heatmap,
        plot_turn_attention_histogram,
        plot_cross_attention_entropy_histogram,
    )

    for i in range(min(5, len(test_results["interview_id"]))):
        plot_cross_attention_heatmap(
            dialogue_id=test_results["interview_id"][i],
            attn_matrix=test_results["cross_attention_weights"][i],
            true_label=test_results["true_label"][i],
            predicted_prob=test_results["probability"][i],
            output_dir=out_dir,
            patient_texts=test_results["utterance_texts"][i] if test_results["utterance_texts"] else None,
            interviewer_texts=test_results["interviewer_utterance_texts"][i] if test_results["interviewer_utterance_texts"] else None,
        )

    plot_turn_attention_histogram(
        [w for w in test_results["turn_attention_weights"]],
        "test", out_dir,
    )

    plot_cross_attention_entropy_histogram(
        test_results["cross_attention_entropy"],
        "test", out_dir,
    )

    # Qualitative Report
    logger.info("Generating cross-attention report...")
    generate_attention_report(test_results, out_dir / "cross_attention_report.md")

    logger.info("Done!")


if __name__ == "__main__":
    main()
