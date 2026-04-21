"""SS-DAMIL-R v18: Lean MIL with Dual Regularization.

Complete redesign combining the two proven regularization strategies:
  - Manifold Mixup at the pooled representation level (from v9d)
  - R-Drop consistency regularization via dual forward passes (from v11b)

Key simplifications from v9d:
  - No symptom auxiliary heads — removes gradient competition
  - No multi-head pooling — single head for stability
  - Sliding-window contextual embeddings (window_size=1 by default)
  - Stronger regularization: dropout=0.4, instance_dropout=0.25, weight_decay=5e-4

Architecture: ~30K training parameters (vs ~50K+ for v9d)
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
from models.ss_damil_r_v18 import SSDamilRClassifierV18
from training.train_damil_r import (
    precompute_dual_role_embeddings,
    set_seed,
)
from training.cv_ss_damil_r_v9a import (
    LabelSmoothingFocalLoss,
    CosineAnnealingWarmRestartsWithWarmup,
    SWACollector,
)
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# R-Drop KL Divergence Loss
# ---------------------------------------------------------------------------

def symmetric_kl_divergence(logits_1: torch.Tensor, logits_2: torch.Tensor) -> torch.Tensor:
    """Compute symmetric KL divergence between two sets of binary logits.

    For binary classification, converts logits to Bernoulli distributions
    and computes (KL(p1||p2) + KL(p2||p1)) / 2.

    Args:
        logits_1: (B,) first set of logits.
        logits_2: (B,) second set of logits.

    Returns:
        Scalar symmetric KL divergence.
    """
    p1 = torch.sigmoid(logits_1)
    p2 = torch.sigmoid(logits_2)

    # Clamp for numerical stability
    p1 = torch.clamp(p1, 1e-7, 1.0 - 1e-7)
    p2 = torch.clamp(p2, 1e-7, 1.0 - 1e-7)

    # KL(p1 || p2) for Bernoulli
    kl_12 = p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))
    # KL(p2 || p1) for Bernoulli
    kl_21 = p2 * torch.log(p2 / p1) + (1 - p2) * torch.log((1 - p2) / (1 - p1))

    return (kl_12 + kl_21).mean() / 2.0


# ---------------------------------------------------------------------------
# Training with Manifold Mixup + R-Drop
# ---------------------------------------------------------------------------

def train_epoch_v18(
    model, loader, main_loss_fn, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0,
    mixup_alpha=0.3,
    rdrop_alpha=0.5,
):
    """Training loop with dual regularization: Manifold Mixup + R-Drop.

    For each batch:
    1. Run backbone → get pooled representations z
    2. Manifold Mixup: mixed z_mix = λ·z + (1-λ)·z[perm], mixed targets
    3. R-Drop: run heads TWICE on z → logit_1, logit_2 (different dropout masks)
    4. Loss = focal(logit_mix, target_mix)
             + 0.5 * focal(logit_1, target)
             + 0.5 * focal(logit_2, target)
             + rdrop_alpha * sym_KL(logit_1, logit_2)
    """
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_rdrop_loss = 0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()

        target = batch["labels"].to(device)

        # Step 1: Run backbone → pooled representations
        pooled_batch = model.forward_batch_split(
            batch["patient_bags"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )

        # Step 2: R-Drop — run heads twice with different dropout masks
        head_out_1 = model.forward_heads(pooled_batch)
        logits_1 = head_out_1["logits"]

        head_out_2 = model.forward_heads(pooled_batch)
        logits_2 = head_out_2["logits"]

        # Average the two logits for tracking metrics (not for loss)
        avg_logits = (logits_1 + logits_2) / 2.0

        # R-Drop: both passes should match the target, and agree with each other
        main_loss_1 = main_loss_fn(logits_1, target)
        main_loss_2 = main_loss_fn(logits_2, target)
        rdrop_loss = symmetric_kl_divergence(logits_1, logits_2)

        combined_main = 0.5 * main_loss_1 + 0.5 * main_loss_2

        # Step 3: Manifold Mixup on pooled representations
        mixup_loss = torch.tensor(0.0, device=device)
        if mixup_alpha > 0 and pooled_batch.size(0) >= 2:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            perm = torch.randperm(pooled_batch.size(0), device=device)

            mixed_pooled = lam * pooled_batch + (1 - lam) * pooled_batch[perm]
            mixed_target = lam * target + (1 - lam) * target[perm]

            mixed_head_out = model.forward_heads(mixed_pooled)
            mixed_logits = mixed_head_out["logits"]
            mixup_loss = main_loss_fn(mixed_logits, mixed_target)

        # Total loss
        loss = combined_main + mixup_loss + rdrop_alpha * rdrop_loss

        if not torch.isfinite(loss):
            logging.warning("  [WARN] Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Check for NaN gradients
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if has_nan_grad:
            logging.warning("  [WARN] NaNs in gradients! Skipping optimizer step.")
            optimizer.zero_grad()
            continue

        optimizer.step()

        total_loss += loss.item()
        total_main_loss += combined_main.item()
        total_rdrop_loss += rdrop_loss.item()

        probs = torch.sigmoid(avg_logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    n_batches = max(1, len(loader))
    metrics = compute_metrics(
        np.array(all_labels),
        (np.array(all_probs) >= 0.5).astype(int),
        np.array(all_probs),
    )
    return (
        total_loss / n_batches,
        total_main_loss / n_batches,
        total_rdrop_loss / n_batches,
        metrics,
    )


# ---------------------------------------------------------------------------
# Evaluation (v18 — no symptom heads)
# ---------------------------------------------------------------------------

def evaluate_v18(model, loader, main_loss_fn, device, threshold=0.5):
    """Evaluate v18 model on a data loader.

    Args:
        model: SSDamilRClassifierV18.
        loader: DataLoader.
        main_loss_fn: Loss function for the main task.
        device: Torch device.
        threshold: Classification threshold.

    Returns:
        Tuple of (avg_loss, main_loss, metrics_dict, predictions_dict).
    """
    model.eval()
    total_loss = 0
    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            output = model.forward_batch(
                batch["patient_bags"].to(device),
                batch["interviewer_bags"].to(device),
                batch["patient_sizes"],
                batch["interviewer_sizes"],
            )
            target = batch["labels"].to(device)
            loss = main_loss_fn(output["logits"], target)
            total_loss += loss.item()

            probs = torch.sigmoid(output["logits"]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    n_batches = max(1, len(loader))
    avg_loss = total_loss / n_batches
    if not np.isfinite(avg_loss):
        avg_loss = 9.999

    return (
        avg_loss,
        avg_loss,  # main_loss = total_loss since no auxiliary heads
        compute_metrics(y_true, y_pred, y_prob),
        {"probability": all_probs, "true_label": all_labels},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SS-DAMIL-R v18: Lean MIL with Dual Regularization"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v18")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config
    parser.add_argument(
        "--encoder_name", type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--att_hidden_dim", type=int, default=32)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=1)

    # Training Config
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--instance_dropout", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--rdrop_alpha_start", type=float, default=1.0)
    parser.add_argument("--rdrop_alpha_end", type=float, default=0.1)
    parser.add_argument("--rdrop_schedule_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)

    # Scheduler Config
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=15)
    parser.add_argument("--cosine_T_mult", type=int, default=2)

    # SWA Config
    parser.add_argument("--swa_checkpoints", type=int, default=5)

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
    logger.info("Loading all interviews with roles...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size

    logger.info(
        f"Pre-computing embeddings (window_size={args.window_size})..."
    )
    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, base_encoder, device,
        max_len=args.max_len,
        window_size=args.window_size,
    )

    # Free encoder memory
    del base_encoder, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _create_model():
        return SSDamilRClassifierV18(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            att_hidden_dim=args.att_hidden_dim,
        ).to(device)

    # Count parameters
    _m = _create_model()
    n_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    logger.info(f"Model trainable parameters: {n_params:,}")
    del _m

    # --- R-Drop alpha schedule ---
    def get_rdrop_alpha(epoch: int) -> float:
        """Linear decay from rdrop_alpha_start → rdrop_alpha_end."""
        progress = min(1.0, epoch / max(1, args.rdrop_schedule_epochs))
        return args.rdrop_alpha_start + progress * (
            args.rdrop_alpha_end - args.rdrop_alpha_start
        )

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        # 1. Subject-level stratified split for internal validation
        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=run_seed
        )
        u_train_idx, u_val_idx = next(
            sss.split(np.zeros(len(u_labels)), u_labels)
        )

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        # --- DATA LEAKAGE ASSERTION ---
        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), \
            "DATA LEAKAGE: train subjects in test set!"
        assert val_sids.isdisjoint(test_sids), \
            "DATA LEAKAGE: val subjects in test set!"
        assert train_sids.isdisjoint(val_sids), \
            "DATA LEAKAGE: train subjects in val set!"

        train_data = [
            iv for iv in train_pool if iv["interview_id"] in train_sids
        ]
        val_data = [
            iv for iv in train_pool if iv["interview_id"] in val_sids
        ]

        train_ds = DualRoleBagDataset(
            train_data, instance_dropout=args.instance_dropout
        )
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dual_role_bags,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_dual_role_bags,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_dual_role_bags,
        )

        # Model
        model = _create_model()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            T_0=args.cosine_T0,
            T_mult=args.cosine_T_mult,
            eta_min=1e-6,
        )

        # Class-weighted Focal Loss
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        neg_to_pos = (len(train_data) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)
        main_loss_fn = LabelSmoothingFocalLoss(
            alpha=alpha, gamma=2.0, smoothing=args.label_smoothing
        )

        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            rdrop_alpha = get_rdrop_alpha(epoch)

            avg_l, ml, rl, _ = train_epoch_v18(
                model, train_loader, main_loss_fn, optimizer, device,
                noise_std=args.noise_std,
                mixup_alpha=args.mixup_alpha,
                rdrop_alpha=rdrop_alpha,
            )
            scheduler.step()

            v_l, _, v_metrics, v_preds = evaluate_v18(
                model, val_loader, main_loss_fn, device
            )

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, R:{rl:.4f}) | "
                    f"Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f} | "
                    f"AUC:{v_metrics.get('roc_auc', 0):.4f} | "
                    f"LR:{current_lr:.2e} | RD-α:{rdrop_alpha:.3f}"
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
                logger.info(
                    f"      Run {run_seed}: Early stopping at epoch {epoch}"
                )
                break

        # Load best and optionally apply SWA
        if not checkpoint_path.exists():
            logger.error(f"      Run {run_seed} FAILED: No checkpoint saved.")
            res_dummy = {
                m: 0.0 for m in [
                    "f1", "accuracy", "precision", "recall",
                    "roc_auc", "balanced_accuracy", "pr_auc",
                ]
            }
            res_dummy["probability"] = [0.5] * len(test_set)
            res_dummy["true_label"] = [iv["label"] for iv in test_set]
            return res_dummy

        if len(swa) >= 3:
            logger.info(
                f"      Run {run_seed}: Applying SWA over "
                f"{len(swa)} checkpoints"
            )
            model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            _, _, best_val_metrics, _ = evaluate_v18(
                model, val_loader, main_loss_fn, device
            )
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v18(
                swa_model, val_loader, main_loss_fn, device
            )
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(
                    f"      SWA improved: ROC-AUC "
                    f"{best_val_score:.4f} -> {swa_val_score:.4f}"
                )
                model = swa_model
            else:
                logger.info(
                    f"      Best checkpoint preferred over SWA "
                    f"({best_val_score:.4f} > {swa_val_score:.4f})"
                )
        else:
            model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )

        # Threshold tuning on validation set
        _, _, v_m, v_p = evaluate_v18(model, val_loader, main_loss_fn, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]),
            np.array(v_p["probability"]),
            metric="loss",
            pos_weight=neg_to_pos,
        )

        # Test evaluation
        _, _, test_metrics, test_results = evaluate_v18(
            model, test_loader, main_loss_fn, device, threshold=best_t
        )
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    logger.info(
        "Starting Stratified Group K-Fold CV "
        "(v18 - Lean MIL + Dual Regularization)..."
    )
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(
            f"# SS-DAMIL-R v18 (Lean MIL + Dual Reg) CV Results\n\n{report}"
        )
    logger.info("\n" + report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump(
            {"aggregate": agg, "raw": raw}, f, indent=4, default=numpy_default
        )

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
