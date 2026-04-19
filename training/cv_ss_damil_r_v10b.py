"""SS-DAMIL-R v10b: MC Dropout Evaluation (ablation).

Uses the same v9d training setup (manifold mixup, dynamic aux, SWA) but
replaces standard eval with Monte Carlo Dropout at inference time:
  - K=10 stochastic forward passes with dropout enabled
  - Average sigmoid probabilities for smoother, better-calibrated estimates
  - Targets improved ROC-AUC and PR-AUC (ranking metrics) and reduced
    cross-run variance
  - Dropout is only in the classifier head (after pooling), so attention
    weights remain deterministic
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
from training.cv_ss_damil_r_v9d import train_epoch_v9d
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# MC Dropout Evaluation
# ---------------------------------------------------------------------------

def evaluate_v9_mcd(model, loader, criterion, device, threshold=0.5, mc_samples=10):
    """Evaluate with Monte Carlo Dropout for better probability calibration.

    Runs K stochastic forward passes with dropout enabled, then averages
    the sigmoid probabilities. This is a principled Bayesian approximation
    that produces smoother probability estimates and reduces variance.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation set.
        criterion: Loss function (used for loss computation, runs in eval mode).
        device: Torch device.
        threshold: Decision threshold for binary predictions.
        mc_samples: Number of stochastic forward passes (K).

    Returns:
        Tuple of (loss, main_loss, metrics_dict, results_dict).
    """
    # Compute loss in standard eval mode (deterministic)
    model.eval()
    total_loss, total_ml = 0, 0
    all_labels, all_ids = [], []

    # First pass: compute loss deterministically
    with torch.no_grad():
        for batch in loader:
            output = model.forward_batch(
                batch["patient_bags"].to(device),
                batch["interviewer_bags"].to(device),
                batch["patient_sizes"],
                batch["interviewer_sizes"],
            )
            target = batch["labels"].to(device)
            loss, ml, _, _ = criterion(
                output["logits"], target,
                output["symptom_logits"], batch["symptoms"].to(device),
                batch["has_symptoms"].to(device),
                diversity_loss=output.get("diversity_loss"),
            )
            total_loss += loss.item()
            total_ml += ml.item()
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

    # Second pass: MC Dropout for probability estimation
    # Enable dropout only (keep batchnorm/layernorm in eval if present)
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.eval()
    model.apply(enable_dropout)

    # Collect K sets of probabilities
    mc_probs = []
    for _ in range(mc_samples):
        sample_probs = []
        with torch.no_grad():
            for batch in loader:
                output = model.forward_batch(
                    batch["patient_bags"].to(device),
                    batch["interviewer_bags"].to(device),
                    batch["patient_sizes"],
                    batch["interviewer_sizes"],
                )
                probs = torch.sigmoid(output["logits"]).cpu().numpy()
                sample_probs.extend(probs)
        mc_probs.append(sample_probs)

    # Average across K samples
    mc_probs = np.array(mc_probs)  # (K, N)
    all_probs = mc_probs.mean(axis=0)  # (N,)

    # Restore full eval mode
    model.eval()

    y_true = np.array(all_labels)
    y_pred = (all_probs >= threshold).astype(int)
    n_batches = max(1, len(loader))
    v_loss = total_loss / n_batches
    v_ml = total_ml / n_batches
    if not np.isfinite(v_loss):
        v_loss = 9.999

    return (
        v_loss, v_ml,
        compute_metrics(y_true, y_pred, all_probs),
        {"probability": all_probs.tolist(), "true_label": all_labels},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v10b: MC Dropout")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v10b")

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

    # Training Config (same as v9d)
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

    # v10b: MC Dropout
    parser.add_argument("--mc_samples", type=int, default=10)

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

        # Same loss as v9d (dynamic aux weight)
        criterion = DynamicMultiTaskDiversityLoss(
            main_loss_fn=main_loss_fn,
            symptom_loss_fn=sym_loss_fn,
            aux_weight_start=args.aux_weight_start,
            aux_weight_end=args.aux_weight_end,
            schedule_epochs=args.aux_schedule_epochs,
            diversity_weight=args.diversity_weight,
        )

        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            criterion.set_epoch(epoch)

            # Training loop: same as v9d (manifold mixup)
            avg_l, ml, al, dl, _ = train_epoch_v9d(
                model, train_loader, criterion, optimizer, device,
                noise_std=args.noise_std,
                mixup_alpha=args.mixup_alpha,
            )
            scheduler.step()

            # v10b: Use MC Dropout for validation too (better threshold tuning)
            v_l, v_ml, v_metrics, v_preds = evaluate_v9_mcd(
                model, val_loader, criterion, device,
                mc_samples=args.mc_samples,
            )

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                aux_w = criterion.current_aux_weight
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}, D:{dl:.4f}) | "
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
            _, _, best_val_metrics, _ = evaluate_v9_mcd(model, val_loader, criterion, device, mc_samples=args.mc_samples)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v9_mcd(swa_model, val_loader, criterion, device, mc_samples=args.mc_samples)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(f"      SWA improved: ROC-AUC {best_val_score:.4f} -> {swa_val_score:.4f}")
                model = swa_model
            else:
                logger.info(f"      Best checkpoint preferred over SWA ({best_val_score:.4f} > {swa_val_score:.4f})")
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # v10b: MC Dropout for threshold tuning and test evaluation
        _, _, v_m, v_p = evaluate_v9_mcd(model, val_loader, criterion, device, mc_samples=args.mc_samples)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v9_mcd(
            model, test_loader, criterion, device,
            threshold=best_t, mc_samples=args.mc_samples,
        )
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV ---
    logger.info("Starting Stratified Group K-Fold CV (v10b - MC Dropout)...")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v10b (MC Dropout K={args.mc_samples}) CV Results\n\n{report}")
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
