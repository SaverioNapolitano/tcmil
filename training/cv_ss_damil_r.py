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

from dataset import (
    load_all_interviews_with_roles,
    DualRoleBagDataset,
    collate_dual_role_bags,
)
from models.ss_damil_r import SSDamilRClassifierV8_2
from training.train_damil_r import (
    precompute_dual_role_embeddings,
    set_seed,
)
from training.train_ss_damil_r import (
    CorrectFocalLoss,
    WeightedSymptomLoss,
    MultiTaskLoss,
    train_epoch,
    evaluate,
    compute_symptom_weights,
)
from utils.evaluation import run_monte_carlo_cv, run_stratified_group_k_fold, run_leave_one_subject_out_cv
from utils.metrics import find_best_threshold
from utils.stats import format_aggregate_report


def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v8.2 Cross-Validation (Recovery)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v8_2")

    # CV Strategy Config
    parser.add_argument("--mode", type=str, default="kfold", choices=["mc", "kfold", "loso"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)

    # Training Config
    parser.add_argument("--dropout_rate", type=float, default=0.3) # Back to baseline
    parser.add_argument("--instance_dropout", type=float, default=0.15)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--aux_weight", type=float, default=0.3, help="Weight for symptom classification loss") # Reduced
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10) # Back to baseline
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss") # More stable

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Setup ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "cv_run.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- Data Loading ---
    logger.info("Loading all interviews with roles and symptoms...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters(): p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size

    logger.info("Pre-computing embeddings...")
    all_interviews = precompute_dual_role_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        # 1. Subject-level stratified split for internal validation
        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

        # Model
        model = SSDamilRClassifierV8_2(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Class Weights for Depression
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        neg_to_pos = (len(train_data) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)

        if args.loss_type == "focal":
            main_loss_fn = CorrectFocalLoss(alpha=alpha, gamma=2.0)
        else:
            p_weight = torch.tensor([neg_to_pos], dtype=torch.float).to(device)
            main_loss_fn = nn.BCEWithLogitsLoss(pos_weight=p_weight)

        # Symptom Class Weights
        sym_weights = compute_symptom_weights(train_data).to(device)
        sym_loss_fn = WeightedSymptomLoss(pos_weight=sym_weights)

        # Multi-Task Criterion
        criterion = MultiTaskLoss(
            main_loss_fn=main_loss_fn,
            symptom_loss_fn=sym_loss_fn,
            aux_weight=args.aux_weight
        )

        # Training Loop
        best_score = -float("inf") if args.checkpoint_metric != "val_loss" else float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            avg_l, ml, al, dl, _ = train_epoch(model, train_loader, criterion, optimizer, device, noise_std=args.noise_std)
            v_l, v_ml, v_metrics, v_preds = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_l)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"      Run {run_seed} Epoch {epoch:02d} | L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}) | Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f}")

            score = v_metrics[args.checkpoint_metric] if args.checkpoint_metric != "val_loss" else v_l
            is_best = (score > best_score) if args.checkpoint_metric != "val_loss" else (score < best_score)

            if is_best:
                best_score = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience: break

        # Load best and evaluate
        if not checkpoint_path.exists():
            logger.error(f"      Run {run_seed} FAILED: No checkpoint saved. Returning zero metrics.")
            res_dummy = {m: 0.0 for m in ["f1", "accuracy", "precision", "recall", "roc_auc", "balanced_accuracy", "loss"]}
            res_dummy["probability"] = [0.5] * len(test_set)
            res_dummy["true_label"] = [iv["label"] for iv in test_set]
            return res_dummy

        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _, _, v_m, v_p = evaluate(model, val_loader, criterion, device)
        best_t = find_best_threshold(np.array(v_p["true_label"]), np.array(v_p["probability"]), metric="loss", pos_weight=neg_to_pos)
        
        _, _, test_metrics, test_results = evaluate(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    if args.mode == "mc":
        agg, raw = run_monte_carlo_cv(all_interviews, train_eval_fn, n_splits=args.n_splits, n_seeds_per_split=args.n_seeds, test_size=0.2, random_state=args.seed)
    elif args.mode == "kfold":
        agg, raw = run_stratified_group_k_fold(all_interviews, train_eval_fn, n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds, random_state=args.seed)
    else:
        agg, raw = run_leave_one_subject_out_cv(all_interviews, train_eval_fn, n_seeds_per_fold=args.n_seeds, random_state=args.seed)

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f: f.write(f"# SS-DAMIL-R (v8.2) CV Results\n\n{report}")
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
