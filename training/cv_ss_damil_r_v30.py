"""SS-DAMIL-R v30: MSD + Uncertainty-Weighted Loss.

Reverts to the stable v9d architecture while introducing:
  - Multi-Sample Dropout (MSD): implicit ensemble on classification heads.
  - Uncertainty-Weighted Multi-Task Learning: automatically balances main, symptom, 
    and diversity loss using homoscedastic uncertainty, removing need for manual scheduling.
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
from models.ss_damil_r import SSDamilRClassifierV30
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
from utils.evaluation import run_stratified_group_k_fold_ensemble
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Learns to balance multiple losses using homoscedastic task uncertainty.
    """
    def __init__(self, main_loss_fn, symptom_loss_fn):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.symptom_loss_fn = symptom_loss_fn
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        msd_logits, targets,
        symptom_logits, symptom_targets, has_symptoms,
        diversity_loss=None,
    ):
        if msd_logits.dim() == 1:
            main_loss = self.main_loss_fn(msd_logits, targets)
        else:
            main_losses = []
            for i in range(msd_logits.size(0)):
                main_losses.append(self.main_loss_fn(msd_logits[i], targets))
            main_loss = torch.stack(main_losses).mean()

        avg_aux_loss = self.symptom_loss_fn(symptom_logits, symptom_targets, has_symptoms)

        div_loss = diversity_loss if diversity_loss is not None else torch.tensor(0.0, device=msd_logits.device)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * main_loss + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * avg_aux_loss + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * div_loss + self.log_vars[2]

        total_loss = loss0 + loss1 + loss2

        return total_loss, main_loss, avg_aux_loss, div_loss


def train_epoch_v30(
    model, loader, criterion, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0,
    mixup_alpha=0.2,
):
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

            mixed_head_output = model.forward_heads(mixed_pooled)
            mixed_msd_logits = mixed_head_output["msd_logits"]
            mixed_logits = mixed_head_output["logits"]
            mixed_sym_logits = mixed_head_output["symptom_logits"]

            mixed_target = lam * target + (1 - lam) * target[perm]
            mixed_sym_target = lam * sym_target + (1 - lam) * sym_target[perm]
            mixed_has_sym = torch.maximum(has_sym, has_sym[perm])

            loss, ml, al, dl = criterion(
                mixed_msd_logits, mixed_target,
                mixed_sym_logits, mixed_sym_target, mixed_has_sym,
                diversity_loss=diversity_loss,
            )
        else:
            loss, ml, al, dl = criterion(
                head_output["msd_logits"], target,
                symptom_logits, sym_target, has_sym,
                diversity_loss=diversity_loss,
            )

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

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    n_batches = max(1, len(loader))
    return (
        total_loss / n_batches, total_m_loss / n_batches,
        total_a_loss / n_batches, total_d_loss / n_batches,
        compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    )


def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v30: MSD + Uncertainty-Weighted Loss")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v30")

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")

    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_pool_heads", type=int, default=2)

    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--instance_dropout", type=float, default=0.20)
    parser.add_argument("--n_msd_samples", type=int, default=5)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke_test:
        args.n_folds = 2
        args.n_seeds = 1
        args.max_epochs = 2
        args.patience = 2
        args.swa_checkpoints = 1

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

    logger.info("Loading all interviews with roles and symptoms...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)
    
    if args.smoke_test:
        all_interviews = all_interviews[:40] # Quick subset for smoke test

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size

    logger.info("Pre-computing embeddings...")
    all_interviews = precompute_dual_role_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)

    def _create_model():
        return SSDamilRClassifierV30(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            n_pool_heads=args.n_pool_heads,
            n_msd_samples=args.n_msd_samples,
        ).to(device)

    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=run_seed)
        u_train_idx, u_val_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))

        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

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

        model = _create_model()

        # Optimizer: notice how we also include the criterion's parameters (log_vars)
        criterion = UncertaintyWeightedMultiTaskLoss(
            main_loss_fn=LabelSmoothingFocalLoss(
                alpha=((len(train_data) - sum(1 for iv in train_data if iv["label"] == 1)) / max(1, sum(1 for iv in train_data if iv["label"] == 1))) / (1 + ((len(train_data) - sum(1 for iv in train_data if iv["label"] == 1)) / max(1, sum(1 for iv in train_data if iv["label"] == 1)))),
                gamma=2.0, smoothing=args.label_smoothing
            ) if args.loss_type == "focal" else nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([(len(train_data) - sum(1 for iv in train_data if iv["label"] == 1)) / max(1, sum(1 for iv in train_data if iv["label"] == 1))], dtype=torch.float).to(device)
            ),
            symptom_loss_fn=WeightedSymptomLoss(pos_weight=compute_symptom_weights(train_data).to(device))
        ).to(device)

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr, weight_decay=1e-4)
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
            avg_l, ml, al, dl, _ = train_epoch_v30(
                model, train_loader, criterion, optimizer, device,
                noise_std=args.noise_std, mixup_alpha=args.mixup_alpha,
            )
            scheduler.step()

            v_l, v_ml, v_metrics, v_preds = evaluate_v9(model, val_loader, criterion, device)

            if epoch % 10 == 0 or epoch == 1 or args.smoke_test:
                current_lr = scheduler.get_last_lr()[0]
                sigmas = torch.exp(criterion.log_vars).detach().cpu().numpy()
                logger.info(
                    f"      Run {run_seed} Epoch {epoch:02d} | "
                    f"L:{avg_l:.4f} (M:{ml:.4f}, A:{al:.4f}, D:{dl:.4f}) | "
                    f"Val L:{v_l:.4f} | F1:{v_metrics['f1']:.4f} | "
                    f"LR:{current_lr:.2e} | Sigmas: {sigmas}"
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
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            _, _, best_val_metrics, _ = evaluate_v9(model, val_loader, criterion, device)
            best_val_score = best_val_metrics.get("roc_auc", 0.0)

            swa_model = _create_model()
            swa.apply(swa_model)
            _, _, swa_val_metrics, _ = evaluate_v9(swa_model, val_loader, criterion, device)
            swa_val_score = swa_val_metrics.get("roc_auc", 0.0)

            if swa_val_score >= best_val_score:
                model = swa_model
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        neg_to_pos = (len(train_data) - num_pos) / max(1, num_pos)
        _, _, v_m, v_p = evaluate_v9(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v9(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {
            **test_metrics, 
            **test_results,
            "val_probability": v_p["probability"],
            "val_true_label": v_p["true_label"]
        }

    logger.info("Starting Stratified Group K-Fold CV (v30 - MSD + Uncertainty Loss)...")
    agg, raw, fold_ensemble_agg = run_stratified_group_k_fold_ensemble(
        all_interviews, train_eval_fn,
        n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    report = format_aggregate_report(agg)
    ensemble_report = format_aggregate_report(fold_ensemble_agg)
    full_report = f"# SS-DAMIL-R v30 (MSD + Uncertainty) CV Results\n\n## Per-Run Aggregated\n{report}\n\n## Fold-Ensemble Aggregated\n{ensemble_report}"
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(full_report)
    logger.info("\n" + full_report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({
            "aggregate": agg, 
            "raw": raw, 
            "fold_ensemble": fold_ensemble_agg
        }, f, indent=4, default=numpy_default)

if __name__ == "__main__":
    main()
