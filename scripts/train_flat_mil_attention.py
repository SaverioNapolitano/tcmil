"""Training script for Flat MIL Attention Pooling Baseline."""

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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Adjust path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews, print_split_stats
from models.flat_mil_attention import FlatMILAttention
from utils.metrics import compute_metrics, confusion_matrix_dict, find_best_threshold, compute_attention_entropy
from utils.plots import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_pr_curve,
    plot_prob_vs_bag_size,
    plot_probability_histogram,
    plot_roc_curve,
    plot_utterance_distribution,
    plot_attention_entropy,
    plot_attention_weights_bar,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BagOfUtterancesDataset(Dataset):
    """Dataset for bags of utterances."""
    def __init__(self, interviews: list[dict]):
        self.interviews = interviews

    def __len__(self) -> int:
        return len(self.interviews)

    def __getitem__(self, idx: int) -> dict:
        return self.interviews[idx]


def build_collate_fn(tokenizer, max_length: int):
    """Build collate function that tokenizes utterances on the fly."""
    def collate_fn(batch: list[dict]) -> dict:
        interview_ids = []
        labels = []
        all_utterances = []
        bag_sizes = []
        utterances_lists = []
        
        for item in batch:
            interview_ids.append(item["interview_id"])
            labels.append(item["label"])
            # Ensure there's at least one utterance (fallback if empty)
            utts = item["utterances"] if item["utterances"] else [""]
            all_utterances.extend(utts)
            bag_sizes.append(len(utts))
            utterances_lists.append(utts)
            
        encoded = tokenizer(
            all_utterances,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "interview_ids": interview_ids,
            "labels": torch.tensor(labels, dtype=torch.float),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "bag_sizes": bag_sizes,
            "utterances_lists": utterances_lists,
        }
    return collate_fn


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, entropy_lambda=0.0):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bag_sizes = batch["bag_sizes"]
        
        optimizer.zero_grad()
        logits, att_weights_list = model(input_ids, attention_mask, bag_sizes)
        loss = criterion(logits, labels)
        
        if entropy_lambda > 0.0:
            # We want to maximize entropy (make attention less peaky), so we subtract entropy from the loss.
            # Entropy = - sum(p * log p)
            # Loss = BCE - lambda * mean_entropy
            entropies = []
            for aw in att_weights_list:
                entropies.append(-torch.sum(aw * torch.log(aw + 1e-9)))
            mean_entropy = torch.stack(entropies).mean()
            loss = loss - entropy_lambda * mean_entropy
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate the model and return loss, metrics, and predictions."""
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bag_sizes = batch["bag_sizes"]
        utterances_lists = batch["utterances_lists"]
        
        logits, att_weights_list = model(input_ids, attention_mask, bag_sizes)
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
    
    # Compute metrics
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


def main():
    parser = argparse.ArgumentParser(description="Train Flat MIL Attention Pooling Baseline")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing preprocessed data")
    parser.add_argument("--output_dir", type=str, default="results/flat_mil_attention", help="Output directory")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained encoder name")
    parser.add_argument("--proj_dim", type=int, default=0, help="Projection dimension (0 to disable)")
    parser.add_argument("--att_hidden_dim", type=int, default=32, help="Attention hidden dimension")
    parser.add_argument("--attention_temp", type=float, default=2.0, help="Temperature for attention softmax")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate before attention scorer")
    parser.add_argument("--entropy_lambda", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (number of bags)")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--max_len", type=int, default=64, help="Max sequence length for each utterance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading interviews...")
    train_data = load_interviews(args.data_dir, "train")
    dev_data = load_interviews(args.data_dir, "dev")
    
    print_split_stats(train_data, "train")
    print_split_stats(dev_data, "dev")
    
    train_dataset = BagOfUtterancesDataset(train_data)
    dev_dataset = BagOfUtterancesDataset(dev_data)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = build_collate_fn(tokenizer, args.max_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- Model Setup ---
    logger.info(f"Initializing model {args.model_name}...")
    proj_dim = args.proj_dim if args.proj_dim > 0 else None
    model = FlatMILAttention(
        args.model_name, 
        proj_dim=proj_dim, 
        att_hidden_dim=args.att_hidden_dim, 
        dropout_rate=args.dropout_rate,
        temperature=args.attention_temp
    )
    model.to(device)
    
    num_pos = sum(1 for iv in train_data if iv["label"] == 1)
    num_neg = len(train_data) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    
    # --- Training Loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_balanced_accuracy": []}
    
    logger.info("Starting training...")
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, entropy_lambda=args.entropy_lambda)
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
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            logger.info("  -> Found new best model (lowest loss): saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break
                
    # Save training history
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    plot_loss_curves(history["train_loss"], history["val_loss"], out_dir)
    plot_metric_curves(history, ["f1", "balanced_accuracy"], out_dir)
    
    # --- Final Evaluation ---
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    
    logger.info("Evaluating on DEV split to tune threshold...")
    _, dev_metrics_default, dev_preds_default = evaluate(model, dev_loader, criterion, device, threshold=0.5)
    best_t = find_best_threshold(
        y_true=np.array(dev_preds_default["true_label"]),
        y_prob=np.array(dev_preds_default["probability"]),
        metric="f1"
    )
    logger.info(f"Best tuned threshold on DEV: {best_t:.4f}")
    
    # Re-evaluate with tuned threshold
    _, dev_metrics, dev_preds = evaluate(model, dev_loader, criterion, device, threshold=best_t)
    
    test_data = load_interviews(args.data_dir, "test")
    print_split_stats(test_data, "test")
    test_dataset = BagOfUtterancesDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    logger.info("Evaluating on TEST split...")
    _, test_metrics, test_preds = evaluate(model, test_loader, criterion, device, threshold=best_t)
    
    # Save metrics
    final_metrics = {"dev": dev_metrics, "test": test_metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Process and save predictions
    dev_df = pd.DataFrame({k: v for k, v in dev_preds.items() if k not in ["attention_weights", "utterance_texts"]})
    dev_df.insert(1, "split", "dev")
    test_df = pd.DataFrame({k: v for k, v in test_preds.items() if k not in ["attention_weights", "utterance_texts"]})
    test_df.insert(1, "split", "test")
    
    all_preds_df = pd.concat([dev_df, test_df], ignore_index=True)
    all_preds_df.to_csv(out_dir / "predictions.csv", index=False)
    
    # Generate Plots
    logger.info("Generating evaluation plots and attention reports...")
    all_attention_records = []
    attention_examples_md = "# Attention Interpretation Examples\n\n"
    
    for split_name, df, metrics_dict, preds_dict in [("dev", dev_df, dev_metrics, dev_preds), ("test", test_df, test_metrics, test_preds)]:
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
                    "utterance_text": t
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
            
    # Qualitative Report (sample from test)
    num_test_samples = min(5, len(test_preds["interview_id"]))
    if num_test_samples > 0:
        sample_indices = np.random.choice(len(test_preds["interview_id"]), num_test_samples, replace=False)
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
                "utterance_texts": texts
            }
            
            # Plot bar chart
            plot_attention_weights_bar(example_dict, out_dir, f"attention_bar_{iv_id}.png")
            
            # Add to Markdown report
            attention_examples_md += f"## Interview: {iv_id}\n"
            attention_examples_md += f"- **True Label**: {true_label}\n"
            attention_examples_md += f"- **Predicted Probability**: {prob:.4f}\n\n"
            attention_examples_md += "### Top 5 Attended Utterances\n"
            
            top_indices = np.argsort(weights)[::-1][:min(5, len(weights))]
            for t_idx in top_indices:
                attention_examples_md += f"**[{t_idx}]** (weight: {weights[t_idx]:.4f}): {texts[t_idx]}\n\n"
                
        with open(out_dir / "attention_examples.md", "w") as f:
            f.write(attention_examples_md)
            
    # Save a small sample table for quick inspection
    sample_df = all_preds_df.sample(min(15, len(all_preds_df)), random_state=args.seed)
    cols_to_print = ["interview_id", "split", "num_utterances", "true_label", "predicted_label", "probability"]
    print("\n--- Sample Predictions ---")
    print(sample_df[cols_to_print].to_markdown(index=False, floatfmt=".4f"))
    
    # Save to a markdown file
    with open(out_dir / "summary.md", "w") as f:
        f.write("# Flat MIL Attention Pooling Baseline\n\n")
        f.write("## Metrics\n```json\n")
        f.write(json.dumps(final_metrics, indent=4))
        f.write("\n```\n\n")
        f.write("## Sample Predictions\n")
        f.write(sample_df[cols_to_print].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
