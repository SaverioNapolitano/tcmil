"""SS-DAMIL-R v16: Set-Transformer + Manifold Mixup + SAM.

Deep redesign combining:
  1. Locality-Aware window context (in precompute_dual_role_embeddings)
  2. Set-Transformer DAMIL-R (Self-Attention without PE replaces BiGRU/Temporal)
  3. Sharpness-Aware Minimization (SAM) combined with Manifold Mixup
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
from models.ss_damil_r import SSDamilRClassifierV16
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
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report
from utils.sam import SAM


# ---------------------------------------------------------------------------
# Training with SAM + Manifold Mixup
# ---------------------------------------------------------------------------

def train_epoch_v16(
    model, loader, criterion, optimizer, device,
    max_grad_norm=1.0, mixup_alpha=0.2,
):
    """Training loop specifically structured for Sharpness-Aware Minimization.
    
    We sample mixup lambdas ONCE per batch, and use them in both the first and 
    second passes of SAM to ensure we optimize the exact same loss landscape.
    """
    model.train()
    total_loss, total_m_loss, total_a_loss, total_d_loss = 0, 0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        target = batch["labels"].to(device)
        sym_target = batch["symptoms"].to(device)
        has_sym = batch["has_symptoms"].to(device)
        
        batch_size = target.size(0)
        
        # Sample Mixup variables ONCE per batch
        if mixup_alpha > 0 and batch_size >= 2:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            perm = torch.randperm(batch_size, device=device)
            mixed_target = lam * target + (1 - lam) * target[perm]
            mixed_sym_target = lam * sym_target + (1 - lam) * sym_target[perm]
            mixed_has_sym = torch.maximum(has_sym, has_sym[perm])
        else:
            lam = None
            perm = None
            mixed_target = target
            mixed_sym_target = sym_target
            mixed_has_sym = has_sym

        first_step_metrics = {}

        def closure():
            optimizer.zero_grad()
            
            # 1. Run backbone (Disable noise so step 1 and step 2 share the same manifold layout)
            pooled_batch, diversity_loss = model.forward_batch_split(
                batch["patient_bags"].to(device),
                batch["interviewer_bags"].to(device),
                batch["patient_sizes"],
                batch["interviewer_sizes"],
                noise_std=0.0, 
            )
            
            # 2. Mixup in manifold
            if lam is not None:
                mixed_pooled = lam * pooled_batch + (1 - lam) * pooled_batch[perm]
                head_output = model.forward_heads(mixed_pooled)
            else:
                head_output = model.forward_heads(pooled_batch)

            # 3. Compute loss
            loss, ml, al, dl = criterion(
                head_output["logits"], mixed_target,
                head_output["symptom_logits"], mixed_sym_target, mixed_has_sym,
                diversity_loss=diversity_loss,
            )
            
            if not torch.isfinite(loss):
                return loss

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            # Save metrics to outer scope if this is the first call (before SAM perturbs weights)
            if "loss" not in first_step_metrics:
                first_step_metrics["loss"] = loss.item()
                first_step_metrics["ml"] = ml.item()
                first_step_metrics["al"] = al.item()
                first_step_metrics["dl"] = dl.item()
                
                # Compute original unmixed predictions for reporting
                with torch.no_grad():
                    orig_head = model.forward_heads(pooled_batch)
                    first_step_metrics["probs"] = torch.sigmoid(orig_head["logits"]).cpu().numpy()
                    
            return loss

        # Run the full 2-step SAM
        loss = optimizer.step(closure)
        
        if not torch.isfinite(loss) or "loss" not in first_step_metrics:
            logging.warning("  [WARN] Non-finite loss detected or closure failed. Skipping batch.")
            optimizer.zero_grad()
            continue

        total_loss += first_step_metrics["loss"]
        total_m_loss += first_step_metrics["ml"]
        total_a_loss += first_step_metrics["al"]
        total_d_loss += first_step_metrics["dl"]
        
        all_probs.extend(first_step_metrics["probs"])
        all_labels.extend(target.cpu().numpy())

    n_batches = max(1, len(loader))
    return (
        total_loss / n_batches, total_m_loss / n_batches,
        total_a_loss / n_batches, total_d_loss / n_batches,
        compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v16: Set-Transformer + SAM")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v16")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Preprocessing Config (v16 feature)
    parser.add_argument("--window_size", type=int, default=1, help="Locality context window size")

    # Model Config (v16 features)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=192) # Increased to handle windowing
    parser.add_argument("--proj_dim", type=int, default=96) # Slightly larger to handle SetTransformer outputs
    parser.add_argument("--n_pool_heads", type=int, default=2)

    # Training Config
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
    
    # SAM Config
    parser.add_argument("--sam_rho", type=float, default=0.05)

    # LR Schedule
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

    logger.info("Pre-computing embeddings with sliding window context...")
    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, base_encoder, device, 
        max_len=args.max_len, window_size=args.window_size
    )

    def _create_model():
        return SSDamilRClassifierV16(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            n_pool_heads=args.n_pool_heads,
        ).to(device)

    # --- Define Training Callback ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        # Leakage Asserts
        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), "DATA LEAKAGE"
        assert val_sids.isdisjoint(test_sids), "DATA LEAKAGE"

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

        model = _create_model()

        # Wrap AdamW with SAM
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, lr=args.lr, weight_decay=1e-4)
        
        # Scheduler applies to base_optimizer inside SAM
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer.base_optimizer,
            warmup_epochs=args.warmup_epochs,
            T_0=args.cosine_T0,
            T_mult=args.cosine_T_mult,
            eta_min=1e-6,
        )

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

        swa = SWACollector(max_checkpoints=args.swa_checkpoints)
        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            criterion.set_epoch(epoch)

            avg_l, ml, al, dl, _ = train_epoch_v16(
                model, train_loader, criterion, optimizer, device,
                mixup_alpha=args.mixup_alpha,
            )
            scheduler.step()

            v_l, v_ml, v_metrics, v_preds = evaluate_v9(model, val_loader, criterion, device)

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
    logger.info("Starting Stratified Group K-Fold CV (v16 - Set-Transformer + SAM)...")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v16 (Set-Transformer + SAM) CV Results\n\n{report}")
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
