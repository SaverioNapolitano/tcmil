"""SS-DAMIL-R v33: Tuned v9d + On-the-fly Text Augmentation + Seed Ensemble.

Combines three improvements over v9d:
  A) Best hyperparameters from grid search (update after running search script)
  B) On-the-fly text augmentation: each epoch, minority-class (depressed)
     training interviews are augmented (word dropout + utterance deletion)
     and re-embedded through the frozen sentence transformer
  C) Seed-ensemble evaluation: probabilities averaged across seeds per fold
"""

import argparse
import copy
import json
import logging
import random
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
    create_augmented_interview,
)
from models.ss_damil_r import SSDamilRClassifierV9
from training.train_damil_r import (
    precompute_dual_role_embeddings,
    mean_pooling,
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
# Lightweight re-embedding for augmented interviews
# ---------------------------------------------------------------------------

@torch.no_grad()
def reembed_interviews(interviews, tokenizer, encoder, device, max_len=128):
    """Re-embed interviews through the frozen sentence transformer.

    Used for on-the-fly augmentation: after modifying utterance text,
    this function produces fresh embedding tensors.
    """
    encoder.eval()
    processed = []
    for iv in interviews:
        # Patient embeddings
        patient_utts = iv.get("utterances", [""])
        if not patient_utts:
            patient_utts = [""]

        encoded_p = tokenizer(
            patient_utts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        ).to(device)
        output_p = encoder(**encoded_p)
        patient_emb = mean_pooling(output_p, encoded_p["attention_mask"])
        patient_emb = F.normalize(patient_emb, p=2, dim=1).cpu()

        # Interviewer embeddings (unmodified text, but re-embed for consistency)
        interviewer_utts = iv.get("interviewer_utterances", [""])
        if not interviewer_utts:
            interviewer_utts = [""]

        encoded_i = tokenizer(
            interviewer_utts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        ).to(device)
        output_i = encoder(**encoded_i)
        interviewer_emb = mean_pooling(output_i, encoded_i["attention_mask"])
        interviewer_emb = F.normalize(interviewer_emb, p=2, dim=1).cpu()

        processed.append({
            **iv,
            "patient_embeddings": patient_emb,
            "interviewer_embeddings": interviewer_emb,
        })
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v33: Tuned + Augmentation + Ensemble")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v33")

    # CV Strategy Config
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    # Model Config (v9)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--n_pool_heads", type=int, default=2)

    # Training Config — defaults are v9d; update after running search
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.20)
    parser.add_argument("--noise_std", type=float, default=0.05)
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

    # v9a training improvements
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

    # v33: Augmentation
    parser.add_argument("--n_augment", type=int, default=2,
                        help="Number of augmented copies per minority-class interview.")
    parser.add_argument("--word_drop_rate", type=float, default=0.05)
    parser.add_argument("--utt_drop_rate", type=float, default=0.2)

    # Smoke test
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 1 fold, 1 seed, 2 epochs for quick verification.")

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

    if args.smoke_test:
        args.max_epochs = 2
        args.patience = 2
        args.n_folds = 2
        args.n_seeds = 1

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

        # Identify minority-class interviews for augmentation
        minority_train = [iv for iv in train_data if iv["label"] == 1]
        logger.info(f"      Run {run_seed}: {len(train_data)} train ({len(minority_train)} minority), "
                    f"augmenting {args.n_augment} copies/epoch")

        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

        # Model
        model = _create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer, warmup_epochs=args.warmup_epochs,
            T_0=args.cosine_T0, T_mult=args.cosine_T_mult, eta_min=1e-6,
        )

        # Original pos ratio for threshold finding later
        orig_num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        orig_neg_to_pos = (len(train_data) - orig_num_pos) / max(1, orig_num_pos)

        # Simulate augmented train data to compute correct loss weights
        train_data_simulated = train_data.copy()
        for iv in minority_train:
            for _ in range(args.n_augment):
                train_data_simulated.append(iv)

        # Loss
        num_pos = sum(1 for iv in train_data_simulated if iv["label"] == 1)
        neg_to_pos = (len(train_data_simulated) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)

        if args.loss_type == "focal":
            main_loss_fn = LabelSmoothingFocalLoss(alpha=alpha, gamma=2.0, smoothing=args.label_smoothing)
        else:
            main_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([neg_to_pos], dtype=torch.float).to(device)
            )

        sym_weights = compute_symptom_weights(train_data_simulated).to(device)
        sym_loss_fn = WeightedSymptomLoss(pos_weight=sym_weights)
        criterion = DynamicMultiTaskDiversityLoss(
            main_loss_fn=main_loss_fn, symptom_loss_fn=sym_loss_fn,
            aux_weight_start=args.aux_weight_start, aux_weight_end=args.aux_weight_end,
            schedule_epochs=args.aux_schedule_epochs, diversity_weight=args.diversity_weight,
        )

        swa = SWACollector(max_checkpoints=args.swa_checkpoints)
        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            criterion.set_epoch(epoch)

            # --- On-the-fly augmentation ---
            augmented = []
            for iv in minority_train:
                for aug_idx in range(args.n_augment):
                    augmented.append(create_augmented_interview(
                        iv, aug_idx,
                        word_drop_rate=args.word_drop_rate,
                        utt_drop_rate=args.utt_drop_rate,
                    ))

            if augmented:
                augmented = reembed_interviews(augmented, tokenizer, base_encoder, device, max_len=args.max_len)

            epoch_train_data = train_data + augmented
            epoch_ds = DualRoleBagDataset(epoch_train_data, instance_dropout=args.instance_dropout)
            epoch_loader = DataLoader(epoch_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)

            # --- Train ---
            avg_l, ml, al, dl, _ = train_epoch_v9d(
                model, epoch_loader, criterion, optimizer, device,
                noise_std=args.noise_std, mixup_alpha=args.mixup_alpha,
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
                    f"LR:{current_lr:.2e} | AuxW:{aux_w:.3f} | Aug:{len(augmented)}"
                )

            score = v_l
            if score < best_score:
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
            metric="loss", pos_weight=orig_neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v9(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {
            **test_metrics, 
            **test_results,
            "val_probability": v_p["probability"],
            "val_true_label": v_p["true_label"]
        }

    # --- Run CV with Seed-Ensemble ---
    logger.info("Starting Stratified Group K-Fold CV (v33 - Tuned + Augmentation + Ensemble)...")
    per_run_agg, raw, fold_ensemble_agg = run_stratified_group_k_fold_ensemble(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # --- Save ---
    per_run_report = format_aggregate_report(per_run_agg)
    ensemble_report = format_aggregate_report(fold_ensemble_agg)

    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v33 (Tuned + Augmentation + Ensemble) CV Results\n\n")
        f.write(f"## Per-Run Metrics\n{per_run_report}\n\n")
        f.write(f"## Seed-Ensemble Metrics (per-fold probability averaging)\n{ensemble_report}\n")
    logger.info("\n--- Per-Run ---\n" + per_run_report)
    logger.info("\n--- Seed-Ensemble ---\n" + ensemble_report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
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
