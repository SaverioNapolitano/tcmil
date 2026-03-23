"""Cross-validation script for Flat MIL Mean Pooling Baseline."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews
from models.flat_mil_mean import FlatMILMeanPooling
from scripts.train_flat_mil_mean import (
    BagOfUtterancesDataset, 
    build_collate_fn, 
    evaluate, 
    set_seed, 
    train_epoch
)
from utils.evaluation import run_monte_carlo_cv
from utils.stats import format_aggregate_report


def make_train_eval_fn(args, device):
    """Factory to create the callback for monte_carlo_cv."""
    def train_eval_fn(train_pool: list[dict], test_set: list[dict], seed: int) -> dict[str, float]:
        set_seed(seed)
        
        # Split train_pool into train (90%) and inner dev (10%) for early stopping
        labels = [iv["label"] for iv in train_pool]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        
        train_data = [train_pool[i] for i in train_idx]
        val_data = [train_pool[i] for i in val_idx]
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        collate_fn = build_collate_fn(tokenizer, args.max_len)
        
        train_loader = DataLoader(
            BagOfUtterancesDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            BagOfUtterancesDataset(val_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            BagOfUtterancesDataset(test_set), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # Initialize model
        proj_dim = args.proj_dim if args.proj_dim > 0 else None
        model = FlatMILMeanPooling(args.model_name, proj_dim=proj_dim)
        model.to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        num_training_steps = len(train_loader) * args.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
        )
        
        best_val_f1 = -1.0
        epochs_no_improve = 0
        best_state = None
        
        for epoch in range(1, args.max_epochs + 1):
            train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            _, val_metrics, _ = evaluate(model, val_loader, criterion, device)
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                # Save best state in memory to avoid messy file io in parallel/loop
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    break
        
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            
        _, test_metrics, _ = evaluate(model, test_loader, criterion, device)
        
        return test_metrics
    return train_eval_fn


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo CV for Flat MIL Mean Pooling")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing preprocessed data")
    parser.add_argument("--output_dir", type=str, default="results/cv_flat_mil_mean", help="Output directory")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny", help="Pretrained encoder name")
    parser.add_argument("--proj_dim", type=int, default=128, help="Projection dimension (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of epochs per run")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--max_len", type=int, default=32, help="Max sequence length for each utterance")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of MC CV splits")
    parser.add_argument("--n_seeds", type=int, default=2, help="Number of seeds per split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load all interviews
    train_data = load_interviews(args.data_dir, "train")
    dev_data = load_interviews(args.data_dir, "dev")
    test_data = load_interviews(args.data_dir, "test")
    all_interviews = train_data + dev_data + test_data
    
    logger.info(f"Loaded {len(all_interviews)} total interviews across all original splits.")
    
    train_eval_fn = make_train_eval_fn(args, device)
    
    # Run Monte Carlo CV
    agg_metrics, all_raw_metrics = run_monte_carlo_cv(
        interviews=all_interviews,
        train_eval_fn=train_eval_fn,
        n_splits=args.n_splits,
        n_seeds_per_split=args.n_seeds,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Format and point report
    report = format_aggregate_report(agg_metrics)
    print("\n--- Final Aggregated Metrics ---")
    print(report)
    
    # Save results
    with open(out_dir / "metrics_cv.json", "w") as f:
        json.dump(agg_metrics, f, indent=4)
        
    with open(out_dir / "raw_metrics_cv.json", "w") as f:
        json.dump(all_raw_metrics, f, indent=4)
        
    with open(out_dir / "cv_summary.md", "w") as f:
        f.write(f"# Cross-Validated Results: Flat MIL Mean\n\n")
        f.write(f"Model: `{args.model_name}`\n")
        f.write(f"Splits: `{args.n_splits}` | Seeds per split: `{args.n_seeds}`\n\n")
        f.write(f"## Aggregated Metrics\n```\n{report}\n```\n")
        
    logger.info(f"Saved CV results to {out_dir}")

if __name__ == "__main__":
    main()
