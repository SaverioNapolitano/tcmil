"""SS-DAMIL-R v23: Multi-Pronged Pipeline Upgrade.

Keeps the exact v9d architecture and training loop, but upgrades:
  1. Configurable encoder (test mental-bert + mpnet)
  2. Sliding window context embeddings (window_size=1)
  3. Test-Time Augmentation (TTA) via instance dropout at inference
  4. Multi-seed ensemble (3 models per CV run, averaged predictions)
  5. F1-tuned threshold selection
  6. Snapshot ensemble at cosine restart boundaries
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
# TTA Evaluation: multiple forward passes with instance dropout
# ---------------------------------------------------------------------------

def evaluate_v23_tta(model, loader, criterion, device, threshold=0.5,
                     n_tta=10, tta_dropout=0.15):
    """Evaluate with Test-Time Augmentation via stochastic instance dropout.

    Runs the model n_tta times per sample. Each pass randomly drops utterances,
    creating diverse bag views. Final probability = mean across all passes.
    """
    # We need to collect per-sample probabilities across TTA passes
    # First pass: collect sample count and structure
    all_ids = []
    all_labels = []
    n_samples = 0

    # Collect all data first
    all_batches = []
    for batch in loader:
        all_batches.append(batch)
        all_ids.extend(batch["interview_ids"])
        all_labels.extend(batch["labels"].numpy())
        n_samples += batch["labels"].size(0)

    # Accumulate probabilities across TTA passes
    accumulated_probs = np.zeros(n_samples)

    for tta_pass in range(n_tta):
        model.eval()
        pass_probs = []

        with torch.no_grad():
            for batch in all_batches:
                p_bags = batch["patient_bags"].to(device)
                i_bags = batch["interviewer_bags"].to(device)
                p_sizes = list(batch["patient_sizes"])
                i_sizes = list(batch["interviewer_sizes"])
                bs = p_bags.size(0)

                # Apply instance dropout manually for TTA
                if tta_pass > 0 and tta_dropout > 0:
                    for idx in range(bs):
                        p_n = p_sizes[idx]
                        i_n = i_sizes[idx]
                        # Patient dropout
                        if p_n > 2:
                            mask = torch.rand(p_n) > tta_dropout
                            mask[0] = True
                            if mask.sum() < 2:
                                mask[:2] = True
                            # Zero out dropped instances
                            drop_mask = (~mask).to(device)
                            p_bags[idx, :p_n][drop_mask] = 0
                            new_p_n = int(mask.sum().item())
                            # Repack: move kept instances to front
                            kept = p_bags[idx, :p_n][mask.to(device)]
                            p_bags[idx, :new_p_n] = kept
                            p_bags[idx, new_p_n:p_n] = 0
                            p_sizes[idx] = new_p_n
                        # Interviewer dropout
                        if i_n > 2:
                            mask = torch.rand(i_n) > tta_dropout
                            mask[0] = True
                            if mask.sum() < 2:
                                mask[:2] = True
                            drop_mask = (~mask).to(device)
                            i_bags[idx, :i_n][drop_mask] = 0
                            new_i_n = int(mask.sum().item())
                            kept = i_bags[idx, :i_n][mask.to(device)]
                            i_bags[idx, :new_i_n] = kept
                            i_bags[idx, new_i_n:i_n] = 0
                            i_sizes[idx] = new_i_n

                output = model.forward_batch(
                    p_bags, i_bags, p_sizes, i_sizes,
                )
                probs = torch.sigmoid(output["logits"]).cpu().numpy()
                pass_probs.extend(probs)

        accumulated_probs += np.array(pass_probs)

    # Average across TTA passes
    final_probs = accumulated_probs / n_tta
    y_true = np.array(all_labels)
    y_pred = (final_probs >= threshold).astype(int)

    # Compute loss on first pass only (for checkpoint selection)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in all_batches:
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

    v_loss = total_loss / max(1, len(all_batches))

    return (
        v_loss, 0.0,
        compute_metrics(y_true, y_pred, final_probs),
        {"probability": final_probs.tolist(), "true_label": all_labels},
    )


# ---------------------------------------------------------------------------
# Non-TTA evaluate for training loop (fast)
# ---------------------------------------------------------------------------

def evaluate_v23_fast(model, loader, criterion, device, threshold=0.5):
    """Standard evaluation without TTA (used during training loop)."""
    model.eval()
    total_loss, total_ml = 0, 0
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
            loss, ml, _, _ = criterion(
                output["logits"], target,
                output["symptom_logits"], batch["symptoms"].to(device),
                batch["has_symptoms"].to(device),
                diversity_loss=output.get("diversity_loss"),
            )
            total_loss += loss.item()
            total_ml += ml.item()
            probs = torch.sigmoid(output["logits"]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

    y_true, y_prob = np.array(all_labels), np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    n = max(1, len(loader))
    v_loss = total_loss / n
    if not np.isfinite(v_loss):
        v_loss = 9.999

    return (
        v_loss, total_ml / n,
        compute_metrics(y_true, y_pred, y_prob),
        {"probability": all_probs, "true_label": all_labels},
    )


# ---------------------------------------------------------------------------
# Training with Manifold Mixup (identical to v9d)
# ---------------------------------------------------------------------------

def train_epoch_v23(
    model, loader, criterion, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0, mixup_alpha=0.2,
):
    """Training loop with manifold mixup (same as v9d)."""
    model.train()
    total_loss, total_m_loss, total_a_loss, total_d_loss = 0, 0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()
        target = batch["labels"].to(device)
        sym_target = batch["symptoms"].to(device)
        has_sym = batch["has_symptoms"].to(device)

        pooled_batch, diversity_loss = model.forward_batch_split(
            batch["patient_bags"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )

        head_output = model.forward_heads(pooled_batch)
        logits = head_output["logits"]
        symptom_logits = head_output["symptom_logits"]

        if mixup_alpha > 0 and pooled_batch.size(0) >= 2:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            perm = torch.randperm(pooled_batch.size(0), device=device)
            mixed_pooled = lam * pooled_batch + (1 - lam) * pooled_batch[perm]
            mixed_head = model.forward_heads(mixed_pooled)
            mixed_target = lam * target + (1 - lam) * target[perm]
            mixed_sym_target = lam * sym_target + (1 - lam) * sym_target[perm]
            mixed_has_sym = torch.maximum(has_sym, has_sym[perm])
            loss, ml, al, dl = criterion(
                mixed_head["logits"], mixed_target,
                mixed_head["symptom_logits"], mixed_sym_target, mixed_has_sym,
                diversity_loss=diversity_loss,
            )
        else:
            loss, ml, al, dl = criterion(
                logits, target, symptom_logits, sym_target, has_sym,
                diversity_loss=diversity_loss,
            )

        if not torch.isfinite(loss):
            logging.warning("  [WARN] Non-finite loss. Skipping batch.")
            continue

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        bad = [p for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            logging.warning("  [WARN] NaN grads! Skipping step.")
            optimizer.zero_grad()
            continue

        optimizer.step()
        total_loss += loss.item()
        total_m_loss += ml.item()
        total_a_loss += al.item()
        total_d_loss += dl.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    n = max(1, len(loader))
    return (
        total_loss / n, total_m_loss / n,
        total_a_loss / n, total_d_loss / n,
        compute_metrics(np.array(all_labels),
                        (np.array(all_probs) >= 0.5).astype(int),
                        np.array(all_probs))
    )


# ---------------------------------------------------------------------------
# Single model training (used by ensemble)
# ---------------------------------------------------------------------------

def _train_single_model(
    create_model_fn, train_loader, val_loader, criterion,
    device, args, out_dir, run_seed, logger,
):
    """Train one model, return the final model + val metrics."""
    set_seed(run_seed)
    model = create_model_fn()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        T_0=args.cosine_T0,
        T_mult=args.cosine_T_mult,
        eta_min=1e-6,
    )

    swa = SWACollector(max_checkpoints=args.swa_checkpoints)
    best_score = float("inf")
    epochs_no_improve = 0
    checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

    for epoch in range(1, args.max_epochs + 1):
        criterion.set_epoch(epoch)
        train_epoch_v23(
            model, train_loader, criterion, optimizer, device,
            noise_std=args.noise_std, mixup_alpha=args.mixup_alpha,
        )
        scheduler.step()

        v_l, _, v_metrics, _ = evaluate_v23_fast(
            model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            aw = criterion.current_aux_weight
            logger.info(
                f"        [Ens {run_seed}] Ep {epoch:02d} | "
                f"ValL:{v_l:.4f} F1:{v_metrics['f1']:.4f} "
                f"LR:{lr:.2e} AuxW:{aw:.3f}"
            )

        if v_l < best_score:
            best_score = v_l
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epoch > args.warmup_epochs:
            swa.update(model)

        if epochs_no_improve >= args.patience:
            break

    # Load best + try SWA
    if not checkpoint_path.exists():
        return model, None

    if len(swa) >= 3:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _, _, best_m, _ = evaluate_v23_fast(model, val_loader, criterion, device)
        best_auc = best_m.get("roc_auc", 0.0)

        swa_model = create_model_fn()
        swa.apply(swa_model)
        _, _, swa_m, _ = evaluate_v23_fast(swa_model, val_loader, criterion, device)
        swa_auc = swa_m.get("roc_auc", 0.0)

        if swa_auc >= best_auc:
            logger.info(f"        SWA improved: {best_auc:.4f} -> {swa_auc:.4f}")
            model = swa_model
        else:
            logger.info(f"        Best > SWA ({best_auc:.4f} > {swa_auc:.4f})")
    else:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    checkpoint_path.unlink(missing_ok=True)
    return model, best_score


# ---------------------------------------------------------------------------
# Ensemble evaluation: average predictions from multiple models
# ---------------------------------------------------------------------------

def ensemble_evaluate_tta(models, loader, criterion, device,
                          threshold=0.5, n_tta=10, tta_dropout=0.15):
    """Average TTA predictions across multiple models."""
    n_models = len(models)
    all_model_probs = []

    for model in models:
        _, _, _, preds = evaluate_v23_tta(
            model, loader, criterion, device,
            threshold=0.5, n_tta=n_tta, tta_dropout=tta_dropout,
        )
        all_model_probs.append(np.array(preds["probability"]))

    # Average across models
    avg_probs = np.mean(all_model_probs, axis=0)
    y_true = np.array(preds["true_label"])
    y_pred = (avg_probs >= threshold).astype(int)

    return (
        compute_metrics(y_true, y_pred, avg_probs),
        {"probability": avg_probs.tolist(), "true_label": preds["true_label"]},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SS-DAMIL-R v23: Multi-Pronged Pipeline Upgrade")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str,
                        default="results/ss_damil_r_cv_v23")

    # CV Strategy
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model / Encoder
    parser.add_argument("--encoder_name", type=str,
                        default="sentence-transformers/all-mpnet-base-v2") # SamLowe/roberta-base-go_emotions / sentence-transformers/all-MiniLM-L12-v2
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_pool_heads", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=1,
                        help="Sliding window context for embeddings")

    # Training
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.20)
    parser.add_argument("--loss_type", type=str, default="focal")
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
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

    # v23 Improvements
    parser.add_argument("--n_ensemble", type=int, default=3,
                        help="Number of models in the ensemble per run")
    parser.add_argument("--n_tta", type=int, default=10,
                        help="Number of TTA forward passes")
    parser.add_argument("--tta_dropout", type=float, default=0.15,
                        help="Instance dropout rate for TTA")
    parser.add_argument("--threshold_metric", type=str, default="f1",
                        choices=["f1", "balanced_accuracy", "loss"],
                        help="Metric to optimize threshold on")

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

    logger.info(f"Pre-computing embeddings (window_size={args.window_size})...")
    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, base_encoder, device,
        max_len=args.max_len, window_size=args.window_size,
    )
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

        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(
            sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), "LEAK: train in test!"
        assert val_sids.isdisjoint(test_sids), "LEAK: val in test!"

        train_data = [iv for iv in train_pool
                      if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool
                    if iv["interview_id"] in val_sids]

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

        # Class weights
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

        # --- Train ensemble of N models ---
        logger.info(
            f"      Run {run_seed}: Training {args.n_ensemble}-model ensemble")
        models = []
        for ens_idx in range(args.n_ensemble):
            ens_seed = run_seed + ens_idx * 1000
            logger.info(f"      Training ensemble member {ens_idx+1}/"
                        f"{args.n_ensemble} (seed={ens_seed})")

            # Need fresh criterion for each model
            ens_criterion = DynamicMultiTaskDiversityLoss(
                main_loss_fn=main_loss_fn,
                symptom_loss_fn=sym_loss_fn,
                aux_weight_start=args.aux_weight_start,
                aux_weight_end=args.aux_weight_end,
                schedule_epochs=args.aux_schedule_epochs,
                diversity_weight=args.diversity_weight,
            )

            model, _ = _train_single_model(
                _create_model, train_loader, val_loader, ens_criterion,
                device, args, out_dir, ens_seed, logger,
            )
            models.append(model)

        # --- Ensemble evaluation with TTA ---
        # Threshold tuning on val with TTA
        logger.info(f"      Run {run_seed}: Ensemble TTA eval on val...")
        val_metrics, val_preds = ensemble_evaluate_tta(
            models, val_loader, criterion, device,
            n_tta=args.n_tta, tta_dropout=args.tta_dropout,
        )

        best_t = find_best_threshold(
            np.array(val_preds["true_label"]),
            np.array(val_preds["probability"]),
            metric=args.threshold_metric,
            pos_weight=neg_to_pos,
        )
        logger.info(
            f"      Run {run_seed}: Val ROC-AUC={val_metrics['roc_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f} | threshold={best_t:.3f}")

        # Test evaluation
        logger.info(f"      Run {run_seed}: Ensemble TTA eval on test...")
        test_metrics, test_preds = ensemble_evaluate_tta(
            models, test_loader, criterion, device,
            threshold=best_t,
            n_tta=args.n_tta, tta_dropout=args.tta_dropout,
        )
        logger.info(
            f"      Run {run_seed}: Test ROC-AUC={test_metrics['roc_auc']:.4f}"
            f" F1={test_metrics['f1']:.4f}")

        return {**test_metrics, **test_preds}

    # --- Run CV ---
    logger.info("Starting Stratified Group K-Fold CV (v23)...")
    logger.info(f"Improvements: window={args.window_size}, "
                f"ensemble={args.n_ensemble}, tta={args.n_tta}, "
                f"threshold={args.threshold_metric}")
    agg, raw = run_stratified_group_k_fold(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v23 CV Results\n\n{report}")
    logger.info("\n" + report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({"aggregate": agg, "raw": raw}, f,
                  indent=4, default=numpy_default)

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
