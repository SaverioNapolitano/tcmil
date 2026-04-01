"""Monte Carlo Cross-Validation for DAMIL-R.

Runs repeated stratified splits at the subject/interview level,
training DAMIL-R from scratch on each fold.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_all_interviews_with_roles
from models.damil_r import DAMILRClassifier
from training.train_damil_r import (
    DualRoleBagDataset,
    collate_dual_role_bags,
    evaluate,
    precompute_dual_role_embeddings,
    set_seed,
    train_epoch,
)
from utils.evaluation import run_monte_carlo_cv
from utils.metrics import find_best_threshold
from utils.stats import format_aggregate_report


def main():
    parser = argparse.ArgumentParser(description="DAMIL-R Monte Carlo Cross-Validation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_r_cv")

    # CV Config
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls"], help="Embedding pooling strategy.")
    parser.add_argument("--proj_dim", type=int, default=64,
                        help="Projection dim before cross-attention. 0 = no projection.")
    parser.add_argument("--att_hidden_dim", type=int, default=32)
    parser.add_argument("--attention_temp", type=float, default=1.0)

    # Training Config
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.15,
                        help="Fraction of utterances to randomly drop during training.")
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss",
                        choices=["val_loss", "f1", "roc_auc", "pr_auc", "balanced_accuracy"])

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
    logger.info("Loading all interviews with roles for CV...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size

    logger.info(f"Pre-computing embeddings (pooling={args.pooling})...")
    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, base_encoder, device,
        max_len=args.max_len, pooling=args.pooling,
    )

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        # Split train_pool into internal train and val
        labels_pool = [iv["label"] for iv in train_pool]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_pool)), labels_pool))

        train_data = [train_pool[i] for i in train_idx]
        val_data = [train_pool[i] for i in val_idx]

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags,
        )

        # Model
        proj_dim = args.proj_dim if args.proj_dim > 0 else None
        model = DAMILRClassifier(
            embedding_dim=embedding_dim,
            proj_dim=proj_dim,
            att_hidden_dim=args.att_hidden_dim,
            dropout_rate=args.dropout_rate,
            temperature=args.attention_temp,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
        )

        # Loss with class weighting
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        num_neg = len(train_data) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training Loop
        best_score = -float("inf") if args.checkpoint_metric != "val_loss" else float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            tr_loss, _ = train_epoch(
                model, train_loader, criterion, optimizer, device,
                entropy_lambda=args.entropy_lambda, max_grad_norm=args.max_grad_norm,
            )
            v_loss, v_metrics, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_loss)

            logger.info(
                f"   [Epoch {epoch:02d}] Loss: {tr_loss:.4f}/{v_loss:.4f} | "
                f"F1: {v_metrics['f1']:.4f} | PR-AUC: {v_metrics['pr_auc']:.4f}"
            )

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

        # Load best and tune threshold
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _, _, v_preds = evaluate(model, val_loader, criterion, device, threshold=0.5)
        best_t = find_best_threshold(
            np.array(v_preds["true_label"]),
            np.array(v_preds["probability"]),
            metric="loss",
            pos_weight=pos_weight.item(),
        )

        # Final test evaluation
        _, test_metrics, _ = evaluate(model, test_loader, criterion, device, threshold=best_t)

        # Cleanup
        checkpoint_path.unlink(missing_ok=True)
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
        f.write("# DAMIL-R Monte Carlo CV Results\n\n")
        f.write(report)

    logger.info("\n" + report)
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
