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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_all_interviews
from models.damil_h import DAMILHClassifier
from training.train_damil_h import (
    EmbeddedBagDataset,
    TokenizedBagDataset,
    collate_embedded_bags,
    collate_tokenized_bags,
    train_epoch,
    evaluate,
    precompute_embeddings,
    set_seed,
    ENCODER_NAME,
    MAX_TOKEN_LENGTH,
    ATT_HIDDEN_DIM,
)
from utils.evaluation import run_monte_carlo_cv
from utils.metrics import find_best_threshold
from utils.stats import format_aggregate_report


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo CV for DAMIL-H")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/cv_damil_h")
    
    # CV Config
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    
    # Model/Training Config
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--unfreeze_top_layers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=0)
    parser.add_argument("--att_hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_metric", type=str, default="pr_auc")
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading all interviews for CV...")
    all_interviews = load_all_interviews(args.data_dir)
    
    is_fine_tuning = (args.unfreeze_top_layers > 0)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    embedding_dim = base_encoder.config.hidden_size
    
    if not is_fine_tuning:
        logger.info(f"Encoder is frozen. Pre-computing embeddings for all data...")
        all_interviews = precompute_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)
        # We can free memory here if not fine-tuning
        # but run_monte_carlo_cv might need it? No, it just passes objects.
        # However, to save VRAM during CV:
        # del base_encoder
        # torch.cuda.empty_cache()

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)
        
        # Split train_pool into internal train and val
        labels_pool = [iv["label"] for iv in train_pool]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_pool)), labels_pool))
        
        train_data = [train_pool[i] for i in train_idx]
        val_data = [train_pool[i] for i in val_idx]
        
        # Decide dataset class and collate
        if is_fine_tuning:
            train_ds = TokenizedBagDataset(train_data, tokenizer, args.max_len)
            val_ds = TokenizedBagDataset(val_data, tokenizer, args.max_len)
            test_ds = TokenizedBagDataset(test_set, tokenizer, args.max_len)
            collate = collate_tokenized_bags
        else:
            train_ds = EmbeddedBagDataset(train_data)
            val_ds = EmbeddedBagDataset(val_data)
            test_ds = EmbeddedBagDataset(test_set)
            collate = collate_embedded_bags
            
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        
        # Model
        # In frozen mode, we use the same base_encoder but it's set to None in model for efficiency
        # In fine-tuning mode, we need it.
        model = DAMILHClassifier(
            encoder=base_encoder if is_fine_tuning else None,
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            att_hidden_dim=args.att_hidden_dim,
        ).to(device)
        
        if is_fine_tuning:
            model.unfreeze_top_n_layers(args.unfreeze_top_layers)
            
        # Optimizer groups
        encoder_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
        param_groups = [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4) # Fixed WD for CV
        
        # Loss
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        num_neg = len(train_data) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training Loop
        best_score = -float("inf") if args.checkpoint_metric != "val_loss" else float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"
        
        for epoch in range(1, args.max_epochs + 1):
            train_epoch(model, train_loader, criterion, optimizer, device, entropy_lambda=args.entropy_lambda, max_grad_norm=args.max_grad_norm, is_tokenized=is_fine_tuning)
            v_loss, v_metrics, _ = evaluate(model, val_loader, criterion, device, is_tokenized=is_fine_tuning)
            
            score = v_metrics[args.checkpoint_metric] if args.checkpoint_metric != "val_loss" else v_loss
            is_best = (score > best_score) if args.checkpoint_metric != "val_loss" else (score < best_score)
            
            if is_best:
                best_score = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    break
                    
        # Load best and Tune
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _, _, v_preds = evaluate(model, val_loader, criterion, device, threshold=0.5, is_tokenized=is_fine_tuning)
        best_t = find_best_threshold(np.array(v_preds["true_label"]), np.array(v_preds["probability"]), metric="loss", pos_weight=pos_weight.item())
        
        # Final test evaluation
        _, test_metrics, _ = evaluate(model, test_loader, criterion, device, threshold=best_t, is_tokenized=is_fine_tuning)
        
        # Cleanup
        checkpoint_path.unlink()
        return test_metrics

    # --- Run CV ---
    agg_metrics, raw_metrics = run_monte_carlo_cv(
        interviews=all_interviews,
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


if __name__ == "__main__":
    main()
