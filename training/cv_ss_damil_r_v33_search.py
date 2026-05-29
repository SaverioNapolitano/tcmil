"""SS-DAMIL-R v33 Hyperparameter Search.

Two-stage grid search over v9d hyperparameters:
  Stage 1: mixup_alpha × lr  (9 configs, 3 seeds/fold)
  Stage 2: Best α/lr from stage 1, vary dropout / instance_dropout / noise_std
           (6 configs, 3 seeds/fold)

Results are saved incrementally so the script can be interrupted and resumed.
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


@torch.no_grad()
def reembed_interviews(interviews, tokenizer, encoder, device, max_len=128):
    """Re-embed interviews through the frozen sentence transformer."""
    encoder.eval()
    processed = []
    for iv in interviews:
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
# Default v9d hyperparameters (baseline)
# ---------------------------------------------------------------------------
V9D_DEFAULTS = dict(
    proj_dim=64,
    n_pool_heads=2,
    dropout_rate=0.3,
    instance_dropout=0.20,
    label_smoothing=0.05,
    aux_weight_start=0.5,
    aux_weight_end=0.1,
    aux_schedule_epochs=40,
    diversity_weight=0.1,
    mixup_alpha=0.2,
    noise_std=0.05,
    batch_size=8,
    max_epochs=80,
    lr=1e-4,
    patience=15,
    warmup_epochs=5,
    cosine_T0=10,
    cosine_T_mult=2,
    swa_checkpoints=5,
    n_augment=2,
    word_drop_rate=0.05,
    utt_drop_rate=0.2,
)


def build_search_configs():
    """Return a list of (name, overrides) for the two-stage grid."""
    configs = []
    # --- Stage 1: mixup_alpha × lr ---
    for ma in [0.1, 0.2, 0.4]:
        for lr in [5e-5, 1e-4, 2e-4]:
            name = f"s1_ma{ma}_lr{lr:.0e}"
            configs.append((name, {"mixup_alpha": ma, "lr": lr}))
    return configs


def build_stage2_configs(best_overrides):
    """Generate stage-2 configs using the best α/lr from stage 1."""
    base = dict(best_overrides)
    configs = []
    for dr in [0.2, 0.4]:
        name = f"s2_dr{dr}"
        configs.append((name, {**base, "dropout_rate": dr}))
    for idp in [0.15, 0.25]:
        name = f"s2_idp{idp}"
        configs.append((name, {**base, "instance_dropout": idp}))
    for ns in [0.03, 0.08]:
        name = f"s2_ns{ns}"
        configs.append((name, {**base, "noise_std": ns}))
    return configs


# ---------------------------------------------------------------------------
# Training callback factory
# ---------------------------------------------------------------------------

def make_train_eval_fn(hp, device, embedding_dim, out_dir, tokenizer, base_encoder, max_len=128, version="v34"):
    """Create a train_eval_fn closure for the given hyperparameters."""

    def _create_model():
        return SSDamilRClassifierV9(
            embedding_dim=embedding_dim,
            proj_dim=hp["proj_dim"],
            dropout_rate=hp["dropout_rate"],
            n_pool_heads=hp["n_pool_heads"],
        ).to(device)

    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        subj_map = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(list(subj_map.keys()))
        u_labels = [subj_map[sid] for sid in u_sids]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=run_seed)
        u_train_idx, u_val_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))
        train_sids = set(u_sids[i] for i in u_train_idx)
        val_sids = set(u_sids[i] for i in u_val_idx)

        test_sids = set(iv["interview_id"] for iv in test_set)
        assert train_sids.isdisjoint(test_sids), "DATA LEAKAGE"
        assert val_sids.isdisjoint(test_sids), "DATA LEAKAGE"

        train_data = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_data = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        minority_train = [iv for iv in train_data if iv["label"] == 1]
        train_data_simulated = train_data.copy()
        
        if version != "v9d":
            for iv in minority_train:
                for _ in range(hp["n_augment"]):
                    train_data_simulated.append(iv)

        val_ds = DualRoleBagDataset(val_data)
        test_ds = DualRoleBagDataset(test_set)

        val_loader = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False, collate_fn=collate_dual_role_bags)
        test_loader = DataLoader(test_ds, batch_size=hp["batch_size"], shuffle=False, collate_fn=collate_dual_role_bags)

        model = _create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer, warmup_epochs=hp["warmup_epochs"],
            T_0=hp["cosine_T0"], T_mult=hp["cosine_T_mult"], eta_min=1e-6,
        )

        orig_num_pos = sum(1 for iv in train_data if iv["label"] == 1)
        orig_neg_to_pos = (len(train_data) - orig_num_pos) / max(1, orig_num_pos)

        num_pos = sum(1 for iv in train_data_simulated if iv["label"] == 1)
        neg_to_pos = (len(train_data_simulated) - num_pos) / max(1, num_pos)
        alpha = neg_to_pos / (1 + neg_to_pos)

        main_loss_fn = LabelSmoothingFocalLoss(alpha=alpha, gamma=2.0, smoothing=hp["label_smoothing"])
        sym_weights = compute_symptom_weights(train_data_simulated).to(device)
        sym_loss_fn = WeightedSymptomLoss(pos_weight=sym_weights)
        criterion = DynamicMultiTaskDiversityLoss(
            main_loss_fn=main_loss_fn, symptom_loss_fn=sym_loss_fn,
            aux_weight_start=hp["aux_weight_start"], aux_weight_end=hp["aux_weight_end"],
            schedule_epochs=hp["aux_schedule_epochs"], diversity_weight=hp["diversity_weight"],
        )

        swa = SWACollector(max_checkpoints=hp["swa_checkpoints"])
        best_score = float("inf")
        epochs_no_improve = 0
        checkpoint_path = out_dir / f"temp_best_{run_seed}.pt"

        for epoch in range(1, hp["max_epochs"] + 1):
            criterion.set_epoch(epoch)
            
            augmented = []
            if version != "v9d":
                for iv in minority_train:
                    for aug_idx in range(hp["n_augment"]):
                        augmented.append(create_augmented_interview(
                            iv, aug_idx,
                            word_drop_rate=hp["word_drop_rate"],
                            utt_drop_rate=hp["utt_drop_rate"],
                        ))

                if augmented:
                    augmented = reembed_interviews(augmented, tokenizer, base_encoder, device, max_len=max_len)

            epoch_train_data = train_data + augmented
            epoch_ds = DualRoleBagDataset(epoch_train_data, instance_dropout=hp["instance_dropout"])
            epoch_loader = DataLoader(epoch_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=collate_dual_role_bags)

            train_epoch_v9d(
                model, epoch_loader, criterion, optimizer, device,
                noise_std=hp["noise_std"], mixup_alpha=hp["mixup_alpha"],
            )
            scheduler.step()
            v_l, _, _, _ = evaluate_v9(model, val_loader, criterion, device)

            if v_l < best_score:
                best_score = v_l
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1

            if epoch > hp["warmup_epochs"]:
                swa.update(model)
            if epochs_no_improve >= hp["patience"]:
                break

        if not checkpoint_path.exists():
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
            if swa_val_metrics.get("roc_auc", 0.0) >= best_val_score:
                model = swa_model
        else:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        _, _, v_m, v_p = evaluate_v9(model, val_loader, criterion, device)
        best_t = find_best_threshold(
            np.array(v_p["true_label"]), np.array(v_p["probability"]),
            metric="loss", pos_weight=orig_neg_to_pos,
        )

        _, _, test_metrics, test_results = evaluate_v9(model, test_loader, criterion, device, threshold=best_t)
        checkpoint_path.unlink(missing_ok=True)
        return {**test_metrics, **test_results}

    return train_eval_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SS-DAMIL-R v33 Hyperparameter Search")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/ss_damil_r_cv_v33_search")
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimization_target", type=str, choices=["ensemble", "single-run"], default="ensemble",
                        help="Optimize search for ensemble performance or individual single-run performance.")
    parser.add_argument("--version", type=str, choices=["v9d", "v33", "v34"], default="v34",
                        help="Which pipeline version to search for. 'v9d' disables text augmentation.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 1 config, 1 fold, 1 seed, 2 epochs for quick verification.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "search.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- Data ---
    logger.info("Loading data and pre-computing embeddings...")
    all_interviews = load_all_interviews_with_roles(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    base_encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad = False
    embedding_dim = base_encoder.config.hidden_size
    all_interviews = precompute_dual_role_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)

    # --- Results file ---
    results_path = out_dir / "search_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    def _numpy_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def run_config(name, overrides):
        if name in all_results:
            logger.info(f"  Skipping {name} (already completed)")
            return
        hp = {**V9D_DEFAULTS, **overrides}

        if args.smoke_test:
            hp["max_epochs"] = 2
            hp["patience"] = 2

        logger.info(f"  Running config: {name}  overrides={overrides}")
        fn = make_train_eval_fn(hp, device, embedding_dim, out_dir, tokenizer, base_encoder, max_len=args.max_len, version=args.version)

        n_folds = 2 if args.smoke_test else args.n_folds
        n_seeds = 1 if args.smoke_test else args.n_seeds

        _, raw, ens_agg = run_stratified_group_k_fold_ensemble(
            all_interviews, fn,
            n_folds=n_folds, n_seeds_per_fold=n_seeds,
            random_state=args.seed,
        )

        # Store both per-run and ensemble metrics
        per_run_auc = float(np.mean([r["roc_auc"] for r in raw]))
        per_run_f1 = float(np.mean([r["f1"] for r in raw]))
        ens_auc = float(ens_agg.get("roc_auc", {}).get("mean", 0.0))
        ens_f1 = float(ens_agg.get("f1", {}).get("mean", 0.0))

        all_results[name] = {
            "overrides": overrides,
            "per_run_auc": per_run_auc,
            "per_run_f1": per_run_f1,
            "ensemble_auc": ens_auc,
            "ensemble_f1": ens_f1,
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_numpy_default)
        logger.info(f"  {name}: run AUC={per_run_auc:.4f} F1={per_run_f1:.4f} | ens AUC={ens_auc:.4f} F1={ens_f1:.4f}")

    # --- Stage 1: mixup_alpha × lr ---
    logger.info("=== Stage 1: mixup_alpha × lr grid ===")
    stage1_configs = build_search_configs()
    if args.smoke_test:
        stage1_configs = stage1_configs[:1]

    for name, overrides in stage1_configs:
        run_config(name, overrides)

    # --- Determine best from stage 1 ---
    stage1_names = [n for n, _ in stage1_configs]
    stage1_results = {n: all_results[n] for n in stage1_names if n in all_results}
    if not stage1_results:
        logger.error("No stage-1 results — cannot proceed to stage 2.")
        return

    target_key = "ensemble_auc" if args.optimization_target == "ensemble" else "per_run_auc"

    best_s1_name = max(stage1_results, key=lambda n: stage1_results[n][target_key])
    best_s1 = stage1_results[best_s1_name]
    logger.info(f"\n=== Stage 1 winner: {best_s1_name} ({target_key}={best_s1[target_key]:.4f}) ===\n")

    # --- Stage 2: regularization refinement ---
    logger.info("=== Stage 2: regularization refinement ===")
    stage2_configs = build_stage2_configs(all_results[best_s1_name]["overrides"])
    if args.smoke_test:
        stage2_configs = stage2_configs[:1]

    for name, overrides in stage2_configs:
        run_config(name, overrides)

    # --- Final ranking ---
    logger.info(f"\n=== Final Ranking (by {target_key}) ===")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1][target_key], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        logger.info(f"  #{rank} {name}: ens AUC={res['ensemble_auc']:.4f} F1={res['ensemble_f1']:.4f} | "
                    f"run AUC={res['per_run_auc']:.4f} F1={res['per_run_f1']:.4f}")

    logger.info(f"\nBest config: {ranked[0][0]} → {ranked[0][1]['overrides']}")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
