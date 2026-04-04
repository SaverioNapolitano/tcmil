"""LoRA-Based Cross-Validation for DAMIL-R (Cluster Edition).

Performs end-to-end fine-tuning of the Transformer encoder via LoRA
across repeated Stratified Group K-Fold or Monte Carlo splits.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure we can import from the project root
sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_all_interviews_with_roles, TokenizedDualRoleBagDataset, collate_lora_bags
from models.damil_r_lora import DAMILRLora
from utils.evaluation import run_monte_carlo_cv, run_stratified_group_k_fold, run_leave_one_subject_out_cv
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="DAMIL-R LoRA Cross-Validation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_r_lora_cv")
    
    # CV Strategy Config
    parser.add_argument("--mode", type=str, default="kfold", choices=["mc", "kfold", "loso"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model/LoRA Config
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Training Config
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

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
    logger.info(f"LoRA-CV Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load All Data
    logger.info("Loading all interviews with roles for LoRA-CV...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    # 2. Define Train/Eval Function for CV
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)
        
        # Internal subject-level val split
        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[idx] for idx in u_train_idx)
        val_sids = set(u_sids[idx] for idx in u_val_idx)

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = TokenizedDualRoleBagDataset(train_data, instance_dropout=0.1)
        val_ds = TokenizedDualRoleBagDataset(val_data)
        test_ds = TokenizedDualRoleBagDataset(test_set)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                                  collate_fn=lambda b: collate_lora_bags(b, tokenizer))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                                collate_fn=lambda b: collate_lora_bags(b, tokenizer))
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                                 collate_fn=lambda b: collate_lora_bags(b, tokenizer))

        # IMPORTANT: Fresh model for each fold/run
        model = DAMILRLora(
            encoder_name=args.encoder_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        ).to(device)

        # Optimization
        lora_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head}
        ], weight_decay=1e-4)

        total_steps = len(train_loader) * args.max_epochs // args.grad_accum
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)

        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        num_neg = len(train_data) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc = 0
        epochs_no_improve = 0
        temp_best_path = out_dir / f"temp_best_{run_seed}.pt"

        # Training Loop
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            for i, batch in enumerate(train_loader):
                p_bag = {k: v.to(device) for k, v in batch["patient_bags"].items()}
                i_bag = {k: v.to(device) for k, v in batch["interviewer_bags"].items()}
                target = batch["labels"].to(device)
                logit, _, _ = model(p_bag, i_bag)
                (criterion(logit, target) / args.grad_accum).backward()
                if (i + 1) % args.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Evaluation on val
            model.eval()
            v_probs, v_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    p_bag = {k: v.to(device) for k, v in batch["patient_bags"].items()}
                    i_bag = {k: v.to(device) for k, v in batch["interviewer_bags"].items()}
                    logit, _, _ = model(p_bag, i_bag)
                    v_probs.append(torch.sigmoid(logit).item())
                    v_labels.append(batch["labels"].item())
            
            v_auc = compute_metrics(np.array(v_labels), (np.array(v_probs) >= 0.5).astype(int), np.array(v_probs))["roc_auc"]
            
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), temp_best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    break

        # Load best and evaluate on test_set
        model.load_state_dict(torch.load(temp_best_path))
        temp_best_path.unlink()

        # Tune threshold on val set for this fold
        t_best_fold = find_best_threshold(np.array(v_labels), np.array(v_probs))

        model.eval()
        t_probs, t_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                p_bag = {k: v.to(device) for k, v in batch["patient_bags"].items()}
                i_bag = {k: v.to(device) for k, v in batch["interviewer_bags"].items()}
                logit, _, _ = model(p_bag, i_bag)
                t_probs.append(torch.sigmoid(logit).item())
                t_labels.append(batch["labels"].item())
        
        y_true = np.array(t_labels)
        y_prob = np.array(t_probs)
        y_pred = (y_prob >= t_best_fold).astype(int)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        return {
            **metrics,
            "true_label": y_true.tolist(),
            "probability": y_prob.tolist()
        }

    # 3. Execution
    if args.mode == "mc":
        agg, raw = run_monte_carlo_cv(all_interviews, train_eval_fn, args.n_splits, args.n_seeds, args.test_size, args.seed)
    elif args.mode == "kfold":
        agg, raw = run_stratified_group_k_fold(all_interviews, train_eval_fn, args.n_folds, args.n_seeds, args.seed)
    else:
        agg, raw = run_leave_one_subject_out_cv(all_interviews, train_eval_fn, args.n_seeds, args.seed)

    # 4. Report
    with open(out_dir / f"{args.mode}_results.json", "w") as f:
        json.dump({"aggregate": agg, "raw": raw}, f, indent=4)
    
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write("# DAMIL-R LoRA CV Final Benchmark\n\n")
        f.write(report)
    
    logger.info("\n" + report)
    logger.info(f"LoRA-CV benchmark completed. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
