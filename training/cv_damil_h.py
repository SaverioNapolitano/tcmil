"""Monte Carlo Cross Validation script for DAMIL-H baseline.

Repeatedly split data into train/dev/test sets to get robust performance estimates.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_all_interviews
from models.damil_h import DAMILHClassifier
from training.train_damil_h import (
    EmbeddedBagDataset,
    collate_embedded_bags,
    train_epoch,
    evaluate,
    precompute_embeddings,
    set_seed,
    ENCODER_NAME,
    MAX_TOKEN_LENGTH,
)
from utils.evaluation import run_monte_carlo_cv
from utils.metrics import find_best_threshold
from utils.stats import format_aggregate_report


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo CV for DAMIL-H")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/cv_damil_h")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_TOKEN_LENGTH)
    parser.add_argument("--proj_dim", type=int, default=0)
    parser.add_argument("--att_hidden_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=50) # Reduced for CV
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--entropy_lambda", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Setup ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(out_dir / "cv_run.log"),
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
    logger.info("Loading all interviews for CV...")
    all_interviews = load_all_interviews(args.data_dir)
    
    # --- Pre-compute embeddings ONCE for all data ---
    logger.info(f"Pre-computing embeddings with frozen {args.encoder_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    all_interviews_embedded = precompute_embeddings(all_interviews, tokenizer, encoder, device)
    embedding_dim = all_interviews_embedded[0]["embeddings"].size(1)
    
    # Free encoder memory
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)
        
        # Split train_pool into internal train and val
        labels_pool = [iv["label"] for iv in train_pool]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_pool)), labels_pool))
        
        train_data = [train_pool[i] for i in train_idx]
        val_data = [train_pool[i] for i in val_idx]
        
        # Dataloaders
        train_loader = DataLoader(
            EmbeddedBagDataset(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_embedded_bags,
        )
        val_loader = DataLoader(
            EmbeddedBagDataset(val_data),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_embedded_bags,
        )
        test_loader = DataLoader(
            EmbeddedBagDataset(test_set),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_embedded_bags,
        )
        
        # Model
        proj_dim = args.proj_dim if args.proj_dim > 0 else None
        model = DAMILHClassifier(
            embedding_dim=embedding_dim,
            proj_dim=proj_dim,
            att_hidden_dim=args.att_hidden_dim,
        ).to(device)
        
        # Loss and Optimizer
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        num_neg = len(train_data) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # Training Loop
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        for epoch in range(1, args.max_epochs + 1):
            train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device, args.entropy_lambda)
            val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Use a temporary file for checkpointing within CV to avoid collisions
                checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    break
                    
        # Load best and evaluate on test
        model.load_state_dict(torch.load(out_dir / f"temp_best_{run_seed}.pt", weights_only=True))
        
        # Tune threshold on val (using weighted loss)
        _, _, val_preds_default = evaluate(model, val_loader, criterion, device, threshold=0.5)
        best_t = find_best_threshold(
            y_true=np.array(val_preds_default["true_label"]),
            y_prob=np.array(val_preds_default["probability"]),
            metric="loss",
            pos_weight=pos_weight.item(),
        )
        
        # Final test evaluation
        _, test_metrics, _ = evaluate(model, test_loader, criterion, device, threshold=best_t)
        
        # Cleanup checkpoint
        (out_dir / f"temp_best_{run_seed}.pt").unlink()
        
        return test_metrics

    # --- Run CV ---
    agg_metrics, raw_metrics = run_monte_carlo_cv(
        interviews=all_interviews_embedded,
        train_eval_fn=train_eval_fn,
        n_splits=args.n_splits,
        n_seeds_per_split=args.n_seeds,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # --- Save Results ---
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump({"aggregate": agg_metrics, "raw": raw_metrics}, f, indent=4)
        
    report = format_aggregate_report(agg_metrics)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write("# DAMIL-H Monte Carlo CV Results\n\n")
        f.write(report)
        
    logger.info("\n" + report)
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
