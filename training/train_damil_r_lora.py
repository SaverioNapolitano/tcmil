"""LoRA Fine-Tuning Script for DAMIL-R (Cluster Edition).

This script performs end-to-end training of the DAMIL-R model, including
Low-Rank Adaptation (LoRA) of the Transformer encoder. Optimized for
high-VRAM GPUs (24GB+) with on-the-fly encoding.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure we can import from the project root
sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews_with_roles, TokenizedDualRoleBagDataset, collate_lora_bags
from models.damil_r_lora import DAMILRLora
from utils.metrics import compute_metrics, find_best_threshold

# Log setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum=8):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        # Move tokenizer outputs to device
        p_bag = {k: v.to(device) for k, v in batch["patient_bags"].items()}
        i_bag = {k: v.to(device) for k, v in batch["interviewer_bags"].items()}
        target = batch["labels"].to(device)

        # Forward
        logit, _, _ = model(p_bag, i_bag)
        loss = criterion(logit, target)
        
        # Backward (scaled by accumulation steps)
        (loss / grad_accum).backward()
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(loader)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            p_bag = {k: v.to(device) for k, v in batch["patient_bags"].items()}
            i_bag = {k: v.to(device) for k, v in batch["interviewer_bags"].items()}
            target = batch["labels"].to(device)

            logit, _, _ = model(p_bag, i_bag)
            prob = torch.sigmoid(logit).item()
            
            all_probs.append(prob)
            all_labels.append(target.item())
            
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    
    return compute_metrics(y_true, y_pred, y_prob), y_true, y_prob


def main():
    parser = argparse.ArgumentParser(description="DAMIL-R LoRA Fine-Tuning")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_r_lora")
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    
    # LoRA Config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Training Hyperparams
    parser.add_argument("--lr_encoder", type=float, default=2e-5, help="LR for LoRA layers")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="LR for DAMIL-R layers")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading tokenized interviews...")
    train_ivs = load_interviews_with_roles(args.data_dir, split="train")
    dev_ivs = load_interviews_with_roles(args.data_dir, split="dev")
    test_ivs = load_interviews_with_roles(args.data_dir, split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    
    train_ds = TokenizedDualRoleBagDataset(train_ivs, instance_dropout=0.1)
    dev_ds = TokenizedDualRoleBagDataset(dev_ivs)
    test_ds = TokenizedDualRoleBagDataset(test_ivs)

    # Note: Using lambda for collate to pass tokenizer cleanly
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                              collate_fn=lambda b: collate_lora_bags(b, tokenizer))
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, 
                            collate_fn=lambda b: collate_lora_bags(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                             collate_fn=lambda b: collate_lora_bags(b, tokenizer))

    # 2. Initialize Model
    model = DAMILRLora(
        encoder_name=args.encoder_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)
    model.print_trainable_parameters()

    # 3. Optimization Setup
    # Differential Learning Rates
    lora_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    
    # Standard Clinical Class Weighting
    num_pos = sum(1 for iv in train_ivs if iv["label"] == 1)
    num_neg = len(train_ivs) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Training Loop
    best_val_auc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.grad_accum)
        
        # Eval on Dev
        dev_metrics, y_true_v, y_prob_v = evaluate(model, dev_loader, device)
        logger.info(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Dev ROC-AUC: {dev_metrics['roc_auc']:.4f}")
        
        if dev_metrics['roc_auc'] > best_val_auc:
            best_val_auc = dev_metrics['roc_auc']
            # Save using PEP-compliant naming
            torch.save(model.state_dict(), out_dir / "best_lora_model.pt")
            logger.info("  -> Best model saved.")

            # Find best threshold on dev for final test report
            best_t = find_best_threshold(y_true_v, y_prob_v)

    # 5. Final Evaluation
    logger.info("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(out_dir / "best_lora_model.pt"))
    test_metrics, _, _ = evaluate(model, test_loader, device, threshold=best_t)
    
    logger.info(f"Final Test Results (Thres={best_t:.2f}):")
    for k, v in test_metrics.items():
        logger.info(f"  {k:15}: {v:.4f}")

    with open(out_dir / "lora_test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()
