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
from tqdm import tqdm
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
# Embedding extraction and Datasets
# ---------------------------------------------------------------------------

ENCODER_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 128  # Default increased to 128
ATT_HIDDEN_DIM = 64
ENCODING_BATCH_SIZE = 64

class EmbeddedBagDataset(Dataset):
    """Dataset for pre-computed utterance embeddings."""
    def __init__(self, interviews: list[dict]):
        self.interviews = interviews

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        item = self.interviews[idx]
        return {
            "bag": item["embeddings"],  # (num_turns, embedding_dim)
            "label": torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
        }

def collate_embedded_bags(batch):
    bags = [item["bag"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["interview_id"] for item in batch]
    
    bag_sizes = [bag.size(0) for bag in bags]
    max_turns = max(bag_sizes)
    embedding_dim = bags[0].size(1)
    
    padded_bags = torch.zeros(len(batch), max_turns, embedding_dim)
    for i, bag in enumerate(bags):
        padded_bags[i, :bag_sizes[i], :] = bag
        
    return {
        "bags": padded_bags,
        "bag_sizes": bag_sizes,
        "labels": labels,
        "interview_ids": ids,
    }

class TokenizedBagDataset(Dataset):
    """Dataset for on-the-fly tokenization (required for fine-tuning)."""
    def __init__(self, interviews: list[dict], tokenizer, max_len: int):
        self.interviews = interviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        item = self.interviews[idx]
        utterances = item["utterances"] if item["utterances"] else [""]
        
        encoded = self.tokenizer(
            utterances,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],  # (num_turns, max_len)
            "attention_mask": encoded["attention_mask"],
            "label": torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
            "utterances": utterances,
        }

def collate_tokenized_bags(batch):
    input_ids_list = [item["input_ids"] for item in batch]
    attr_mask_list = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["interview_id"] for item in batch]
    utts = [item["utterances"] for item in batch]
    
    bag_sizes = [ids_bag.size(0) for ids_bag in input_ids_list]
    max_turns = max(bag_sizes)
    max_len = input_ids_list[0].size(1)
    
    padded_ids = torch.zeros(len(batch), max_turns, max_len, dtype=torch.long)
    padded_masks = torch.zeros(len(batch), max_turns, max_len, dtype=torch.long)
    
    for i in range(len(batch)):
        padded_ids[i, :bag_sizes[i], :] = input_ids_list[i]
        padded_masks[i, :bag_sizes[i], :] = attr_mask_list[i]
        
    return {
        "bags": padded_ids,
        "attention_masks": padded_masks,
        "bag_sizes": bag_sizes,
        "labels": labels,
        "interview_ids": ids,
        "utterances_lists": utts,
    }

@torch.no_grad()
def precompute_embeddings(interviews, tokenizer, encoder, device, max_len=MAX_TOKEN_LENGTH):
    encoder.eval()
    processed = []
    
    for iv in tqdm(interviews, desc="Pre-computing embeddings"):
        utts = iv.get("utterances", [""])
        if not utts:
            utts = [""]
        
        encoded = tokenizer(
            utts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        
        outputs = encoder(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        
        processed.append({**iv, "embeddings": embeddings})
            
    return processed

# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, entropy_lambda=0.0, max_grad_norm=1.0, is_tokenized=False):
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        optimizer.zero_grad()
        target = batch["labels"].to(device)
        bag_sizes = batch["bag_sizes"]
        
        if is_tokenized:
            bags = batch["bags"].to(device)
            attn_masks = batch["attention_masks"].to(device)
            logits, att_weights_list = model.forward_batch(
                bags, bag_sizes, is_tokenized=True, attention_masks=attn_masks
            )
        else:
            bags = batch["bags"].to(device)
            logits, att_weights_list = model.forward_batch(bags, bag_sizes)

        loss = criterion(logits, target)

        if entropy_lambda > 0:
            entropy = 0
            for attW in att_weights_list:
                entropy -= torch.sum(attW * torch.log(attW + 1e-9))
            loss = loss - entropy_lambda * (entropy / len(att_weights_list))

        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    return avg_loss, metrics


def evaluate(model, loader, criterion, device, threshold=0.5, is_tokenized=False):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_ids = []
    all_att_weights = []
    all_utterances = []

    with torch.no_grad():
        for batch in loader:
            target = batch["labels"].to(device)
            bag_sizes = batch["bag_sizes"]
            
            if is_tokenized:
                bags = batch["bags"].to(device)
                attn_masks = batch["attention_masks"].to(device)
                logits, att_weights_list = model.forward_batch(
                    bags, bag_sizes, is_tokenized=True, attention_masks=attn_masks
                )
                all_utterances.extend(batch["utterances_lists"])
            else:
                bags = batch["bags"].to(device)
                logits, att_weights_list = model.forward_batch(bags, bag_sizes)
                # In non-tokenized mode, the batch doesn't strictly need utterances unless we plot
                if "utterances_lists" in batch:
                    all_utterances.extend(batch["utterances_lists"])

            loss = criterion(logits, target)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])
            all_att_weights.extend([w.cpu().numpy() for w in att_weights_list])

    avg_loss = total_loss / len(loader)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    raw_preds = {
        "interview_id": all_ids,
        "true_label": all_labels,
        "probability": all_probs,
        "attention": all_att_weights,
        "utterances": all_utterances if all_utterances else None
    }
    
    return avg_loss, metrics, raw_preds


def main():
    parser = argparse.ArgumentParser(description="Train DAMIL-H for Depression Detection")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_h")
    
    # Model Config
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_TOKEN_LENGTH)
    parser.add_argument("--proj_dim", type=int, default=0)
    parser.add_argument("--att_hidden_dim", type=int, default=64)
    parser.add_argument("--attention_temp", type=float, default=1.0)
    
    # Training Config
    parser.add_argument("--unfreeze_top_layers", type=int, default=0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_metric", type=str, default="pr_auc", choices=["val_loss", "f1", "roc_auc", "pr_auc", "balanced_accuracy"])
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading interviews...")
    train_ivs = load_interviews(args.data_dir, split="train")
    dev_ivs = load_interviews(args.data_dir, split="dev")
    test_ivs = load_interviews(args.data_dir, split="test")

    is_fine_tuning = (args.unfreeze_top_layers > 0)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    embedding_dim = base_encoder.config.hidden_size
    
    model = DAMILHClassifier(
        encoder=base_encoder,
        embedding_dim=embedding_dim,
        proj_dim=args.proj_dim,
        att_hidden_dim=args.att_hidden_dim,
        dropout_rate=args.dropout_rate,
        temperature=args.attention_temp,
    ).to(device)
    
    if is_fine_tuning:
        logger.info(f"Unfreezing top {args.unfreeze_top_layers} layers of the encoder...")
        model.unfreeze_top_n_layers(args.unfreeze_top_layers)
        train_dataset = TokenizedBagDataset(train_ivs, tokenizer, args.max_len)
        dev_dataset = TokenizedBagDataset(dev_ivs, tokenizer, args.max_len)
        test_dataset = TokenizedBagDataset(test_ivs, tokenizer, args.max_len)
        collate_fn = collate_tokenized_bags
    else:
        logger.info(f"Encoder is frozen. Pre-computing embeddings...")
        train_ivs_emb = precompute_embeddings(train_ivs, tokenizer, base_encoder, device, max_len=args.max_len)
        dev_ivs_emb = precompute_embeddings(dev_ivs, tokenizer, base_encoder, device, max_len=args.max_len)
        test_ivs_emb = precompute_embeddings(test_ivs, tokenizer, base_encoder, device, max_len=args.max_len)
        model.encoder = None
        train_dataset = EmbeddedBagDataset(train_ivs_emb)
        dev_dataset = EmbeddedBagDataset(dev_ivs_emb)
        test_dataset = EmbeddedBagDataset(test_ivs_emb)
        collate_fn = collate_embedded_bags

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer groups
    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
    param_groups = [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    num_pos = sum(1 for iv in train_ivs if iv["label"] == 1)
    num_neg = len(train_ivs) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training Loop ---
    logger.info("Starting training...")
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_bacc": [], "val_roc_auc": [], "val_pr_auc": []}
    best_score = -float("inf") if args.checkpoint_metric != "val_loss" else float("inf")
    epochs_no_improve = 0
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, tr_metrics = train_epoch(model, train_loader, criterion, optimizer, device, entropy_lambda=args.entropy_lambda, max_grad_norm=args.max_grad_norm, is_tokenized=is_fine_tuning)
        v_loss, v_metrics, _ = evaluate(model, dev_loader, criterion, device, is_tokenized=is_fine_tuning)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_metrics["f1"])
        history["val_bacc"].append(v_metrics["balanced_accuracy"])
        history["val_roc_auc"].append(v_metrics["roc_auc"])
        history["val_pr_auc"].append(v_metrics["pr_auc"])

        logger.info(f"Epoch {epoch:02d} | Loss: {tr_loss:.4f}/{v_loss:.4f} | PR-AUC: {v_metrics['pr_auc']:.4f} | ROC-AUC: {v_metrics['roc_auc']:.4f}")

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

    # Final Eval
    if is_fine_tuning: model.encoder = base_encoder
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    _, _, v_preds = evaluate(model, dev_loader, criterion, device, threshold=0.5, is_tokenized=is_fine_tuning)
    best_t = find_best_threshold(np.array(v_preds["true_label"]), np.array(v_preds["probability"]), metric="loss", pos_weight=pos_weight.item())
    logger.info(f"Best threshold: {best_t:.4f}")

    _, test_metrics, test_results = evaluate(model, test_loader, criterion, device, threshold=best_t, is_tokenized=is_fine_tuning)
    with open(out_dir / "metrics.json", "w") as f: json.dump({"test": test_metrics, "threshold": best_t}, f, indent=4)
    
    # --- Plotting ---
    logger.info("Generating evaluation plots...")
    plot_loss_curves(history["train_loss"], history["val_loss"], out_dir)
    plot_metric_curves(history, ["f1", "balanced_accuracy"], out_dir)
    plot_roc_curve(np.array(test_results["true_label"]), np.array(test_results["probability"]), "test", out_dir)
    plot_confusion_matrix(np.array(test_results["true_label"]), (np.array(test_results["probability"]) >= best_t).astype(int), "test", out_dir)
    
    # Custom attention plot
    for i in range(min(5, len(test_results["interview_id"]))):
        sample_id = test_results["interview_id"][i]
        example = {
            "interview_id": sample_id,
            "attention_weights": test_results["attention"][i],
            "true_label": int(test_results["true_label"][i]),
            "probability": float(test_results["probability"][i]),
            "utterance_texts": test_results["utterances"][i] if test_results["utterances"] else None
        }
        plot_attention_weights_bar(example, out_dir, f"attention_bar_{sample_id}.png")

    logger.info("Done!")

if __name__ == "__main__":
    main()
