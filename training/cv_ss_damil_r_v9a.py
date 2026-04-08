"""SS-DAMIL-R v9a: Training Loop Improvements (ablation).

Uses the same v8.2 model architecture but with improved training:
  - Cosine Annealing with Warm Restarts (T₀=10, T_mult=2)
  - Linear warmup (5 epochs)
  - Stochastic Weight Averaging (SWA) over last 5 checkpoints
  - Label smoothing (ε=0.05) for the main task
  - Increased max_epochs=80, patience=15
  - 5 seeds/fold for tighter CIs
"""

import argparse
import copy
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
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# Label Smoothing
# ---------------------------------------------------------------------------

class LabelSmoothingFocalLoss(nn.Module):
    """Focal Loss with label smoothing applied to targets."""
    def __init__(self, alpha=0.5, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.focal = CorrectFocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.focal(inputs, targets)


# ---------------------------------------------------------------------------
# LR Schedule with Warmup
# ---------------------------------------------------------------------------

class CosineAnnealingWarmRestartsWithWarmup:
    """Wraps CosineAnnealingWarmRestarts with a linear warmup phase."""

    def __init__(self, optimizer, warmup_epochs, T_0, T_mult, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        # Store base LRs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.warmup_start_lrs = [lr / 10.0 for lr in self.base_lrs]

        # The cosine scheduler (will be used after warmup)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / max(1, self.warmup_epochs)
            for pg, start_lr, base_lr in zip(
                self.optimizer.param_groups, self.warmup_start_lrs, self.base_lrs
            ):
                pg['lr'] = start_lr + progress * (base_lr - start_lr)
        else:
            # Cosine phase: reset the cosine scheduler epoch counter
            cosine_epoch = self.current_epoch - self.warmup_epochs
            self.cosine_scheduler.step(cosine_epoch)

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Stochastic Weight Averaging
# ---------------------------------------------------------------------------

class SWACollector:
    """Collects model state dicts for Stochastic Weight Averaging."""

    def __init__(self, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.state_dicts = []

    def update(self, model: nn.Module):
        sd = copy.deepcopy(model.state_dict())
        self.state_dicts.append(sd)
        if len(self.state_dicts) > self.max_checkpoints:
            self.state_dicts.pop(0)

    def apply(self, model: nn.Module):
        """Average all collected state dicts and load into model."""
        if not self.state_dicts:
            return
        avg_sd = {}
        for key in self.state_dicts[0]:
            stacked = torch.stack([sd[key].float() for sd in self.state_dicts])
            avg_sd[key] = stacked.mean(dim=0)
        model.load_state_dict(avg_sd)

    def __len__(self):
        return len(self.state_dicts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v9a: Training Loop Improvements")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v9a")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config (same as v8.2)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)

    # Training Config (IMPROVED)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.15)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss")

    # v9a-specific
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

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
    for p in base_encoder.parameters():
        p.requires_grad = False
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

        # --- DATA LEAKAGE ASSERTION ---
        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), "DATA LEAKAGE: train subjects in test set!"
        assert val_sids.isdisjoint(test_sids), "DATA LEAKAGE: val subjects in test set!"
        # --------------------------------

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

        # Model (same v8.2 architecture)
        model = SSDamilRClassifierV8_2(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # v9a: Cosine Annealing with Warm Restarts + Warmup
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            T_0=args.cosine_T0,
            T_mult=args.cosine_T_mult,
            eta_min=1e-6,
        )

        # Class Weights for Depression
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        neg_to_pos = (len(train_data) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)

        # v9a: Label Smoothing Focal Loss
        if args.loss_type == "focal":
            main_loss_fn = LabelSmoothingFocalLoss(alpha=alpha, gamma=2.0, smoothing=args.label_smoothing)
        else:
            main_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([neg_to_pos], dtype=torch.float).to(device)
            )

        # Symptom Class Weights
        sym_weights = compute_symptom_weights(train_data).to(device)
        sym_loss_fn = WeightedSymptomLoss(pos_weight=sym_weights)

        # Multi-Task Criterion
        criterion = MultiTaskLoss(
            main_loss_fn=main_loss_fn,
            symptom_loss_fn=sym_loss_fn,
            aux_weight=args.aux_weight,
        )

        # v9a: SWA Collector
        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        # Training Loop
        best_score = float("inf")  # val_loss mode
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            avg_l, ml, al, dl, _ = train_epoch(
                model, train_loader, criterion, optimizer, device, noise_std=args.noise_std
            )

            # v9a: Step the cosine+warmup scheduler (epoch-based)
            scheduler.step()

            v_l, v_ml, v_metrics, v_preds = evaluate(model, val_loader, criterion, device)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}) | "
                    f"Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f} | "
                    f"LR:{current_lr:.2e}"
                )

            score = v_l  # checkpoint_metric = val_loss
            is_best = score < best_score

            if is_best:
                best_score = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1

            # v9a: Collect SWA checkpoint every epoch after warmup
            if epoch > args.warmup_epochs:
                swa.update(model)

            if epochs_no_improve >= args.patience:
                break

        # Load best and apply SWA
        if not checkpoint_path.exists():
            logger.error(f"      Run {run_seed} FAILED: No checkpoint saved.")
            res_dummy = {m: 0.0 for m in ["f1", "accuracy", "precision", "recall", "roc_auc", "balanced_accuracy", "pr_auc"]}
            res_dummy["probability"] = [0.5] * len(test_set)
            res_dummy["true_label"] = [iv["label"] for iv in test_set]
            return res_dummy

        # v9a: Apply SWA if we have enough checkpoints
        if len(swa) >= 3:
            logger.info(f"      Run {run_seed}: Applying SWA over {len(swa)} checkpoints")
            # Load best, then try SWA – use whichever is better on val
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            _, _, best_val_metrics, _ = evaluate(model, val_loader, criterion, device)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = SSDamilRClassifierV8_2(
                embedding_dim=embedding_dim,
                proj_dim=args.proj_dim,
                dropout_rate=args.dropout_rate,
            ).to(device)
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate(swa_model, val_loader, criterion, device)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(f"      SWA improved: ROC-AUC {best_val_score:.4f} -> {swa_val_score:.4f}")
                model = swa_model
            else:
                logger.info(f"      Best checkpoint preferred over SWA ({best_val_score:.4f} > {swa_val_score:.4f})")
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Threshold tuning on val
        _, _, v_m, v_p = evaluate(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos
        )

        # Test evaluation
        _, _, test_metrics, test_results = evaluate(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    logger.info("Starting Stratified Group K-Fold CV (v9a)...")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v9a (Training Loop Improvements) CV Results\n\n{report}")
    logger.info("\n" + report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({"aggregate": agg, "raw": raw}, f, indent=4, default=numpy_default)

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
