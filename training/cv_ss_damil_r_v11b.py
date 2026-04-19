"""SS-DAMIL-R v11b: R-Drop Consistency Regularization.

Targets the high run-to-run variance (ROC-AUC std ~0.08) by forcing
prediction consistency across two stochastic forward passes of the
same input. Each batch runs through the model twice with dropout
enabled, and a symmetric KL-divergence term penalizes disagreements.

This smooths the loss landscape, reduces sensitivity to initialization,
and improves generalization — all without adding parameters.

Uses the v9 model architecture with v9a training improvements.
Dropout increased to 0.4 so the two passes are meaningfully different.
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
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import (
    load_all_interviews_with_roles,
    DualRoleBagDataset,
    collate_dual_role_bags,
)
from models.ss_damil_r import SSDamilRClassifierV9
from training.train_damil_r import (
    precompute_dual_role_embeddings,
    set_seed,
)
from training.train_ss_damil_r import (
    WeightedSymptomLoss,
    compute_symptom_weights,
)
from training.cv_ss_damil_r_v9a import (
    LabelSmoothingFocalLoss,
    CosineAnnealingWarmRestartsWithWarmup,
    SWACollector,
)
from training.cv_ss_damil_r_v9b import (
    MultiTaskDiversityLoss,
    evaluate_v9,
)
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# R-Drop Loss
# ---------------------------------------------------------------------------

class RDropLoss(nn.Module):
    """R-Drop: Regularized Dropout for Neural Networks.

    Minimizes the bidirectional KL divergence between two stochastic
    forward passes of the same input. For binary classification,
    this operates on the Bernoulli distributions defined by the
    sigmoid outputs.

    Reference: Wu et al. "R-Drop: Regularized Dropout for Neural Networks"
               (NeurIPS 2021)

    Args:
        alpha: Weight for the KL divergence term.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor) -> torch.Tensor:
        """Compute symmetric KL divergence between two sets of predictions.

        Args:
            logits_1: Logits from first forward pass, shape (B,).
            logits_2: Logits from second forward pass, shape (B,).

        Returns:
            Scalar symmetric KL divergence loss.
        """
        p1 = torch.sigmoid(logits_1)
        p2 = torch.sigmoid(logits_2)

        # Clamp for numerical stability
        eps = 1e-7
        p1 = torch.clamp(p1, eps, 1 - eps)
        p2 = torch.clamp(p2, eps, 1 - eps)

        # KL(p1 || p2) for Bernoulli distributions
        kl_12 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
        # KL(p2 || p1)
        kl_21 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())

        # Symmetric KL
        return self.alpha * (kl_12 + kl_21).mean() / 2


# ---------------------------------------------------------------------------
# Training with R-Drop
# ---------------------------------------------------------------------------

def train_epoch_v11b(
    model, loader, criterion, rdrop_loss_fn, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0,
):
    """Training loop with R-Drop consistency regularization.

    Each batch runs through the model TWICE (both with dropout enabled).
    The classification loss is averaged over both passes, and a KL
    divergence term penalizes prediction disagreements.
    """
    model.train()
    total_loss, total_m_loss, total_a_loss, total_d_loss, total_r_loss = 0, 0, 0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()

        target = batch["labels"].to(device)
        sym_target = batch["symptoms"].to(device)
        has_sym = batch["has_symptoms"].to(device)

        patient_bags = batch["patient_bags"].to(device)
        interviewer_bags = batch["interviewer_bags"].to(device)
        p_sizes = batch["patient_sizes"]
        i_sizes = batch["interviewer_sizes"]

        # Forward pass 1 (with dropout stochasticity)
        output_1 = model.forward_batch(
            patient_bags, interviewer_bags, p_sizes, i_sizes,
            noise_std=noise_std,
        )

        # Forward pass 2 (same input, different dropout mask)
        output_2 = model.forward_batch(
            patient_bags, interviewer_bags, p_sizes, i_sizes,
            noise_std=noise_std,
        )

        # Classification loss on both passes (averaged)
        loss_1, ml_1, al_1, dl_1 = criterion(
            output_1["logits"], target,
            output_1["symptom_logits"], sym_target, has_sym,
            diversity_loss=output_1.get("diversity_loss"),
        )

        loss_2, ml_2, al_2, dl_2 = criterion(
            output_2["logits"], target,
            output_2["symptom_logits"], sym_target, has_sym,
            diversity_loss=output_2.get("diversity_loss"),
        )

        cls_loss = (loss_1 + loss_2) / 2
        ml = (ml_1 + ml_2) / 2
        al = (al_1 + al_2) / 2
        dl = (dl_1 + dl_2) / 2

        # R-Drop consistency loss
        rdrop_loss = rdrop_loss_fn(output_1["logits"], output_2["logits"])

        # Total loss
        loss = cls_loss + rdrop_loss

        if not torch.isfinite(loss):
            logging.warning("  [WARN] Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        params_with_nan = [p for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if params_with_nan:
            logging.warning("  [WARN] NaNs in gradients! Skipping optimizer step.")
            optimizer.zero_grad()
            continue

        optimizer.step()

        total_loss += loss.item()
        total_m_loss += ml.item()
        total_a_loss += al.item()
        total_d_loss += dl.item()
        total_r_loss += rdrop_loss.item()

        # Track metrics on pass 1
        probs = torch.sigmoid(output_1["logits"]).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    n_batches = max(1, len(loader))
    return (
        total_loss / n_batches, total_m_loss / n_batches,
        total_a_loss / n_batches, total_d_loss / n_batches,
        total_r_loss / n_batches,
        compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v11b: R-Drop Consistency")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v11b")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config (v9)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_pool_heads", type=int, default=2)

    # Training Config
    parser.add_argument("--dropout_rate", type=float, default=0.4)  # INCREASED for meaningful R-Drop
    parser.add_argument("--instance_dropout", type=float, default=0.15)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss")

    # v11b-specific
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)

    # v9a training improvements
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

    def _create_model():
        return SSDamilRClassifierV9(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            n_pool_heads=args.n_pool_heads,
        ).to(device)

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        # 1. Subject-level stratified split
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

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

        # Model (v9 architecture, higher dropout)
        model = _create_model()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

        if args.loss_type == "focal":
            main_loss_fn = LabelSmoothingFocalLoss(alpha=alpha, gamma=2.0, smoothing=args.label_smoothing)
        else:
            main_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([neg_to_pos], dtype=torch.float).to(device)
            )

        sym_weights = compute_symptom_weights(train_data).to(device)
        sym_loss_fn = WeightedSymptomLoss(pos_weight=sym_weights)

        criterion = MultiTaskDiversityLoss(
            main_loss_fn=main_loss_fn,
            symptom_loss_fn=sym_loss_fn,
            aux_weight=args.aux_weight,
            diversity_weight=args.diversity_weight,
        )

        # v11b: R-Drop Loss
        rdrop_loss_fn = RDropLoss(alpha=args.rdrop_alpha)

        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            avg_l, ml, al, dl, rl, _ = train_epoch_v11b(
                model, train_loader, criterion, rdrop_loss_fn, optimizer, device,
                noise_std=args.noise_std,
            )
            scheduler.step()

            v_l, v_ml, v_metrics, v_preds = evaluate_v9(model, val_loader, criterion, device)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}, D:{dl:.4f}, R:{rl:.4f}) | "
                    f"Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f} | "
                    f"LR:{current_lr:.2e}"
                )

            score = v_l
            is_best = score < best_score

            if is_best:
                best_score = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1

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

        if len(swa) >= 3:
            logger.info(f"      Run {run_seed}: Applying SWA over {len(swa)} checkpoints")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            _, _, best_val_metrics, _ = evaluate_v9(model, val_loader, criterion, device)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v9(swa_model, val_loader, criterion, device)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(f"      SWA improved: ROC-AUC {best_val_score:.4f} -> {swa_val_score:.4f}")
                model = swa_model
            else:
                logger.info(f"      Best checkpoint preferred over SWA ({best_val_score:.4f} > {swa_val_score:.4f})")
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        _, _, v_m, v_p = evaluate_v9(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v9(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    logger.info("Starting Stratified Group K-Fold CV (v11b - R-Drop)...")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v11b (R-Drop Consistency) CV Results\n\n{report}")
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
