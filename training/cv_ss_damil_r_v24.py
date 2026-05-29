"""SS-DAMIL-R v24: Optimised Training Pipeline (No Architectural Changes).

Keeps the EXACT v9d model architecture (SSDamilRClassifierV9) and training
loop (manifold mixup), but refines the training protocol:

  1. Discriminative weight decay: heavier regularisation on the projector
     (most prone to overfitting), lighter on attention/injection layers.
  2. Extended SWA: 10 checkpoints, collection starts from epoch 3.
  3. Longer patience (20) and max_epochs (100) to let cosine cycles complete.
  4. Higher instance dropout (0.25) for more diverse bag views.
  5. Fold-aware seed-ensemble aggregation: averages predictions across the
     5 seeds within each fold before computing metrics.
  6. Threshold tuning via metric="loss" (proven in v9d baseline).
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
from training.cv_ss_damil_r_v9b import evaluate_v9
from training.cv_ss_damil_r_v9c import DynamicMultiTaskDiversityLoss
from training.cv_ss_damil_r_v9d import train_epoch_v9d
from utils.evaluation import run_stratified_group_k_fold_ensemble
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# Discriminative Weight Decay: parameter group builder
# ---------------------------------------------------------------------------

def build_param_groups(model, lr, wd_projector=5e-4, wd_default=1e-4, wd_light=1e-5):
    """Build parameter groups with discriminative weight decay.

    - Projector layers: high WD (prevents overfitting on the projection)
    - Attention pooling, symptom injection, classifier: low WD (need flexibility)
    - Everything else (cross-attention, fusion, norms): default WD
    """
    projector_params = []
    light_params = []
    default_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "projector" in name and "sym_projector" not in name:
            projector_params.append(param)
        elif any(k in name for k in [
            "pooling", "sym_projector", "inject_gate",
            "main_classifier", "symptom_head",
        ]):
            light_params.append(param)
        else:
            default_params.append(param)

    return [
        {"params": projector_params, "lr": lr, "weight_decay": wd_projector},
        {"params": default_params, "lr": lr, "weight_decay": wd_default},
        {"params": light_params, "lr": lr, "weight_decay": wd_light},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SS-DAMIL-R v24: Optimised Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str,
                        default="results/ss_damil_r_cv_v24")

    # CV Strategy
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model (identical to v9d)
    parser.add_argument("--encoder_name", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_pool_heads", type=int, default=2)

    # Training (v24 tuned defaults)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.25)  # UP from 0.20
    parser.add_argument("--loss_type", type=str, default="focal")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_weight_start", type=float, default=0.5)
    parser.add_argument("--aux_weight_end", type=float, default=0.1)
    parser.add_argument("--aux_schedule_epochs", type=int, default=40)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)         # UP from 80
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)            # UP from 15
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=10)     # UP from 5
    parser.add_argument("--swa_start_epoch", type=int, default=3)      # DOWN from warmup

    # Weight decay per group
    parser.add_argument("--wd_projector", type=float, default=5e-4)
    parser.add_argument("--wd_default", type=float, default=1e-4)
    parser.add_argument("--wd_light", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "cv_run.log"),
                  logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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
    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)
    logger.info(f"Encoder: {args.encoder_name}, dim={embedding_dim}")

    def _create_model():
        return SSDamilRClassifierV9(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            n_pool_heads=args.n_pool_heads,
        ).to(device)

    # --- Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        # 1. Subject-level stratified split for internal validation
        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(
            sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        # --- DATA LEAKAGE ASSERTION ---
        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), "DATA LEAKAGE: train subjects in test set!"
        assert val_sids.isdisjoint(test_sids), "DATA LEAKAGE: val subjects in test set!"

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = DualRoleBagDataset(
            train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_dual_role_bags)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_dual_role_bags)

        # Model (exact v9d architecture)
        model = _create_model()

        # v24: Discriminative weight decay
        param_groups = build_param_groups(
            model, lr=args.lr,
            wd_projector=args.wd_projector,
            wd_default=args.wd_default,
            wd_light=args.wd_light,
        )
        optimizer = torch.optim.AdamW(param_groups)

        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            T_0=args.cosine_T0,
            T_mult=args.cosine_T_mult,
            eta_min=1e-6,
        )

        # Class weights for depression
        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        neg_to_pos = (len(train_data) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)

        main_loss_fn = LabelSmoothingFocalLoss(
            alpha=alpha, gamma=2.0, smoothing=args.label_smoothing)

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

        # v24: Extended SWA
        swa = SWACollector(max_checkpoints=args.swa_checkpoints)

        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            criterion.set_epoch(epoch)

            avg_l, ml, al, dl, _ = train_epoch_v9d(
                model, train_loader, criterion, optimizer, device,
                noise_std=args.noise_std,
                mixup_alpha=args.mixup_alpha,
            )
            scheduler.step()

            v_l, v_ml, v_metrics, v_preds = evaluate_v9(
                model, val_loader, criterion, device)

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

            # v24: Start SWA collection earlier (epoch 3 vs epoch 5)
            if epoch > args.swa_start_epoch:
                swa.update(model)

            if epochs_no_improve >= args.patience:
                break

        # Load best and apply SWA
        if not checkpoint_path.exists():
            logger.error(f"      Run {run_seed} FAILED: No checkpoint saved.")
            res_dummy = {m: 0.0 for m in [
                "f1", "accuracy", "precision", "recall",
                "roc_auc", "balanced_accuracy", "pr_auc"]}
            res_dummy["probability"] = [0.5] * len(test_set)
            res_dummy["true_label"] = [iv["label"] for iv in test_set]
            return res_dummy

        if len(swa) >= 3:
            logger.info(f"      Run {run_seed}: Applying SWA over {len(swa)} checkpoints")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            _, _, best_val_metrics, _ = evaluate_v9(
                model, val_loader, criterion, device)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v9(
                swa_model, val_loader, criterion, device)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                logger.info(
                    f"      SWA improved: ROC-AUC "
                    f"{best_val_score:.4f} -> {swa_val_score:.4f}")
                model = swa_model
            else:
                logger.info(
                    f"      Best checkpoint preferred over SWA "
                    f"({best_val_score:.4f} > {swa_val_score:.4f})")
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Threshold tuning on val (metric="loss" — proven in v9d)
        _, _, v_m, v_p = evaluate_v9(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos,
        )

        # Test evaluation
        _, _, test_metrics, test_results = evaluate_v9(
            model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    # --- Run CV with seed-ensemble aggregation ---
    logger.info("Starting Stratified Group K-Fold CV (v24 — Optimised Pipeline)...")
    logger.info(f"Changes vs v9d: instance_dropout={args.instance_dropout}, "
                f"patience={args.patience}, max_epochs={args.max_epochs}, "
                f"swa_ckpts={args.swa_checkpoints}, swa_start={args.swa_start_epoch}, "
                f"wd_proj={args.wd_projector}, wd_light={args.wd_light}")

    per_run_agg, raw, fold_ensemble_agg = run_stratified_group_k_fold_ensemble(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    per_run_report = format_aggregate_report(per_run_agg)
    fold_ens_report = format_aggregate_report(fold_ensemble_agg)

    full_report = (
        f"# SS-DAMIL-R v24 CV Results\n\n"
        f"## Per-Run Aggregation (25 runs)\n\n{per_run_report}\n\n"
        f"## Fold-Ensemble Aggregation (5 fold-level ensembles)\n\n{fold_ens_report}"
    )

    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(full_report)
    logger.info("\n" + full_report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({
            "per_run_aggregate": per_run_agg,
            "fold_ensemble_aggregate": fold_ensemble_agg,
            "raw": raw,
        }, f, indent=4, default=numpy_default)

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
