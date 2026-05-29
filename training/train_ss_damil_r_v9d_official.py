"""SS-DAMIL-R v9d (Official Splits): Manifold Mixup (ablation).

Trains on the official DAIC-WOZ train split, tunes on dev, and evaluates on test.
Supports multiple seeds for robust evaluation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import (
    load_interviews_with_roles,
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
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import compute_aggregate_metrics, format_aggregate_report

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v9d: Official Splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_official_v9d")

    # Official Strategy Config
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds to run")

    # Model Config (v9)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.05)
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

    # v9a training improvements
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_T_mult", type=int, default=2)
    parser.add_argument("--swa_checkpoints", type=int, default=5)

    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    # --- Setup ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "official_run.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- Data Loading ---
    logger.info("Loading official train, dev, and test splits...")
    train_data_raw = load_interviews_with_roles(args.data_dir, "train")
    dev_data_raw = load_interviews_with_roles(args.data_dir, "dev")
    test_data_raw = load_interviews_with_roles(args.data_dir, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size

    logger.info("Pre-computing embeddings for all splits...")
    all_data_raw = train_data_raw + dev_data_raw + test_data_raw
    all_data_embedded = precompute_dual_role_embeddings(all_data_raw, tokenizer, base_encoder, device, max_len=args.max_len)

    # Split them back
    train_data = all_data_embedded[:len(train_data_raw)]
    dev_data = all_data_embedded[len(train_data_raw):len(train_data_raw)+len(dev_data_raw)]
    test_data = all_data_embedded[len(train_data_raw)+len(dev_data_raw):]
    
    logger.info(f"Train size: {len(train_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}")

    def _create_model():
        return SSDamilRClassifierV9(
            embedding_dim=embedding_dim,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            n_pool_heads=args.n_pool_heads,
        ).to(device)

    all_raw_metrics = []

    # Prepare dev and test loaders since they don't change across seeds
    dev_ds = DualRoleBagDataset(dev_data)
    test_ds = DualRoleBagDataset(test_data)
    val_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_role_bags)

    for seed_idx in range(args.n_seeds):
        run_seed = args.base_seed + seed_idx
        logger.info(f"\n{'='*50}\nStarting Run {seed_idx+1}/{args.n_seeds} with seed {run_seed}\n{'='*50}")
        set_seed(run_seed)

        train_ds = DualRoleBagDataset(train_data, instance_dropout=args.instance_dropout)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_role_bags)

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

        # Class Weights for Depression (computed on train split ONLY)
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

        # v9d: Same dynamic aux weight as v9c
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
            # Update dynamic aux weight
            criterion.set_epoch(epoch)

            avg_l, ml, al, dl, _ = train_epoch_v9d(
                model, train_loader, criterion, optimizer, device,
                noise_std=args.noise_std,
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

        # Load best and apply SWA
        if not checkpoint_path.exists():
            logger.error(f"      Run {run_seed} FAILED: No checkpoint saved.")
            res_dummy = {m: 0.0 for m in ["f1", "accuracy", "precision", "recall", "roc_auc", "balanced_accuracy", "pr_auc"]}
            res_dummy["probability"] = [0.5] * len(test_data)
            res_dummy["true_label"] = [iv["label"] for iv in test_data]
            all_raw_metrics.append(res_dummy)
            continue

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

        # Tune threshold on DEV set
        _, _, v_m, v_p = evaluate_v9(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=neg_to_pos,
        )

        # Evaluate on TEST set
        _, _, test_metrics, test_results = evaluate_v9(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        
        run_res = {**test_metrics, **test_results}
        run_res["_seed_idx"] = seed_idx
        run_res["_run_seed"] = run_seed
        all_raw_metrics.append(run_res)
        logger.info(f"      Run {run_seed} Test F1: {test_metrics['f1']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # --- Aggregate and Save ---
    logger.info("Aggregating results across all seeds...")
    agg = compute_aggregate_metrics(all_raw_metrics)
    report = format_aggregate_report(agg)
    
    with open(out_dir / "official_report.txt", "w") as f:
        f.write(f"# SS-DAMIL-R v9d (Official Splits) Results\n\n{report}")
    logger.info("\n" + report)

    def numpy_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "official_results.json", "w") as f:
        json.dump({"aggregate": agg, "raw": all_raw_metrics}, f, indent=4, default=numpy_default)

    logger.info(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
