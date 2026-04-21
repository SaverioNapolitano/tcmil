"""SS-DAMIL-R v19: Deeply Improving Over v9d Baseline.

Combines the EXACT v9d architecture (multi-head pooling + gated symptom injection)
with three orthogonal improvements:
1. Manifold Mixup (from v9d) - Augments data in representation space.
2. R-Drop Consistency Regularization (from v11b) - Smooths the loss landscape.
3. Monte Carlo Dropout at evaluation - Reduces variance and improves accuracy.
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
from training.cv_ss_damil_r_v9c import DynamicMultiTaskDiversityLoss
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# R-Drop Loss
# ---------------------------------------------------------------------------

class RDropLoss(nn.Module):
    """R-Drop: Regularized Dropout for Neural Networks."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor) -> torch.Tensor:
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
# Training with Manifold Mixup + R-Drop
# ---------------------------------------------------------------------------

def train_epoch_v19(
    model, loader, criterion, rdrop_loss_fn, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0,
    mixup_alpha=0.2,
):
    model.train()
    total_loss, total_m_loss, total_a_loss, total_d_loss, total_r_loss = 0, 0, 0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()

        target = batch["labels"].to(device)
        sym_target = batch["symptoms"].to(device)
        has_sym = batch["has_symptoms"].to(device)

        # 1. Run backbone only to get pooled representations
        pooled_batch, diversity_loss = model.forward_batch_split(
            batch["patient_bags"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )

        # 2. Manifold Mixup — mix pooled representations
        if mixup_alpha > 0 and pooled_batch.size(0) >= 2:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            perm = torch.randperm(pooled_batch.size(0), device=device)

            mixed_pooled = lam * pooled_batch + (1 - lam) * pooled_batch[perm]

            mixed_head_output = model.forward_heads(mixed_pooled)

            mixed_target = lam * target + (1 - lam) * target[perm]
            mixed_sym_target = lam * sym_target + (1 - lam) * sym_target[perm]
            mixed_has_sym = torch.maximum(has_sym, has_sym[perm])

            L_mix, mix_ml, mix_al, mix_dl = criterion(
                mixed_head_output["logits"], mixed_target,
                mixed_head_output["symptom_logits"], mixed_sym_target, mixed_has_sym,
                diversity_loss=diversity_loss,
            )
        else:
            # Fallback if batch size < 2
            head_output = model.forward_heads(pooled_batch)
            L_mix, mix_ml, mix_al, mix_dl = criterion(
                head_output["logits"], target,
                head_output["symptom_logits"], sym_target, has_sym,
                diversity_loss=diversity_loss,
            )

        # 3. R-Drop Consistency — run unmixed representations TWICE
        out_1 = model.forward_heads(pooled_batch)
        out_2 = model.forward_heads(pooled_batch)

        L_1, ml_1, al_1, dl_1 = criterion(
            out_1["logits"], target,
            out_1["symptom_logits"], sym_target, has_sym,
            diversity_loss=diversity_loss,
        )

        L_2, ml_2, al_2, dl_2 = criterion(
            out_2["logits"], target,
            out_2["symptom_logits"], sym_target, has_sym,
            diversity_loss=diversity_loss,
        )

        R_kl = rdrop_loss_fn(out_1["logits"], out_2["logits"])

        # 4. Total Loss
        loss = L_mix + 0.5 * (L_1 + L_2) + R_kl

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
        total_m_loss += (mix_ml + 0.5 * (ml_1 + ml_2)).item()
        total_a_loss += (mix_al + 0.5 * (al_1 + al_2)).item()
        total_d_loss += diversity_loss.item()
        total_r_loss += R_kl.item()

        probs = torch.sigmoid(out_1["logits"]).detach().cpu().numpy()
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
# Evaluation with MC Dropout
# ---------------------------------------------------------------------------

def evaluate_v19(model, loader, criterion, device, threshold=0.5, mc_passes=1):
    model.eval()

    # Enable dropout modules if using MC Dropout
    if mc_passes > 1:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    total_loss, total_ml = 0, 0
    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            target = batch["labels"].to(device)
            sym_target = batch["symptoms"].to(device)
            has_sym = batch["has_symptoms"].to(device)

            p_bags = batch["patient_bags"].to(device)
            i_bags = batch["interviewer_bags"].to(device)
            p_sizes = batch["patient_sizes"]
            i_sizes = batch["interviewer_sizes"]

            if mc_passes > 1:
                batch_probs_sum = 0
                for _ in range(mc_passes):
                    out = model.forward_batch(p_bags, i_bags, p_sizes, i_sizes)
                    batch_probs_sum += torch.sigmoid(out["logits"])

                loss, ml, _, _ = criterion(
                    out["logits"], target,
                    out["symptom_logits"], sym_target, has_sym,
                    diversity_loss=out.get("diversity_loss"),
                )
                mean_probs = (batch_probs_sum / mc_passes).cpu().numpy()
            else:
                out = model.forward_batch(p_bags, i_bags, p_sizes, i_sizes)
                loss, ml, _, _ = criterion(
                    out["logits"], target,
                    out["symptom_logits"], sym_target, has_sym,
                    diversity_loss=out.get("diversity_loss"),
                )
                mean_probs = torch.sigmoid(out["logits"]).cpu().numpy()

            total_loss += loss.item()
            total_ml += ml.item()

            all_probs.extend(mean_probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

    y_true, y_prob = np.array(all_labels), np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    n_batches = max(1, len(loader))
    v_loss = total_loss / n_batches
    v_ml = total_ml / n_batches
    if not np.isfinite(v_loss):
        v_loss = 9.999

    return (
        v_loss, v_ml,
        compute_metrics(y_true, y_pred, y_prob),
        {"probability": all_probs, "true_label": all_labels},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v19: Deeply Improving Over v9d")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v19")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config (v9d standard)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_pool_heads", type=int, default=2)

    # Training Config (from v9d)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.20)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_weight_start", type=float, default=0.5)
    parser.add_argument("--aux_weight_end", type=float, default=0.1)
    parser.add_argument("--aux_schedule_epochs", type=int, default=40)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss")

    # v9a training improvements
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

    # NEW: v19 additions
    parser.add_argument("--rdrop_alpha", type=float, default=0.5)
    parser.add_argument("--mc_passes", type=int, default=10)
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "balanced_accuracy", "loss"])

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

        # Model (v9 architecture)
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

        criterion = DynamicMultiTaskDiversityLoss(
            main_loss_fn=main_loss_fn,
            symptom_loss_fn=sym_loss_fn,
            aux_weight_start=args.aux_weight_start,
            aux_weight_end=args.aux_weight_end,
            schedule_epochs=args.aux_schedule_epochs,
            diversity_weight=args.diversity_weight,
        )

        rdrop_loss_fn = RDropLoss(alpha=args.rdrop_alpha)
        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            criterion.set_epoch(epoch)

            avg_l, ml, al, dl, rl, _ = train_epoch_v19(
                model, train_loader, criterion, rdrop_loss_fn, optimizer, device,
                noise_std=args.noise_std,
                mixup_alpha=args.mixup_alpha,
            )
            scheduler.step()

            # Note: For efficiency, we don't use MC dropout during epoch-level val evaluation
            v_l, v_ml, v_metrics, v_preds = evaluate_v19(model, val_loader, criterion, device, mc_passes=1)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                aux_w = criterion.current_aux_weight
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}, D:{dl:.4f}, R:{rl:.4f}) | "
                    f"Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f} | "
                    f"LR:{current_lr:.2e} | AuxW:{aux_w:.3f}"
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
            _, _, best_val_metrics, _ = evaluate_v19(model, val_loader, criterion, device, mc_passes=1)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v19(swa_model, val_loader, criterion, device, mc_passes=1)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(f"      SWA improved: ROC-AUC {best_val_score:.4f} -> {swa_val_score:.4f}")
                model = swa_model
            else:
                logger.info(f"      Best checkpoint preferred over SWA ({best_val_score:.4f} > {swa_val_score:.4f})")
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Full MC Dropout evaluation for threshold tuning and test
        _, _, v_m, v_p = evaluate_v19(model, val_loader, criterion, device, mc_passes=args.mc_passes)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric=args.threshold_metric, pos_weight=neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v19(model, test_loader, criterion, device, threshold=best_t, mc_passes=args.mc_passes)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    logger.info("Starting Stratified Group K-Fold CV (v19 - Deeply Improving Over v9d)...")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v19 (Deeply Improving Over v9d) CV Results\n\n{report}")
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
