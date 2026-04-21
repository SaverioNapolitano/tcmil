"""DAMIL-CL: Stratified Group K-Fold CV with Clinical Linguistic Features.

Improvements over the DAMIL-R / SS-DAMIL-R baselines:

  1. Dual-stream instances  — frozen MPNet embeddings + 16 clinical linguistic
     features (disfluency, pronouns, negation, affect, position, …).
  2. ABMIL with gating      — more selective instance weighting than plain tanh
     attention.
  3. Multi-view bag repr    — attention-pool ‖ mean-pool (complementary views).
  4. R-Drop regularization  — two forward passes per batch, symmetric KL penalty
     forces prediction consistency under different dropout masks.
  5. Stochastic Weight Avg  — weights collected in the last 25 % of training and
     averaged before final evaluation (flattens the loss landscape).
  6. Cosine LR w/ warmup    — smoother optimization than ReduceOnPlateau.
  7. Label smoothing        — ε = 0.05, prevents overconfidence on tiny dataset.
  8. Leakage-free ling norm — StandardScaler fitted on training utterances only,
     applied to val/test without seeing their statistics.
"""

import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_all_interviews_with_roles
from extract_linguistic_features import (
    N_FEATURES,
    extract_all_ling_features,
    normalize_ling_features,
)
from models.damil_cl import DAMILCLClassifier
from training.train_damil_r import (
    precompute_dual_role_embeddings,
    set_seed,
)
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# Dataset and collation (extends dual-role bags with linguistic features)
# ---------------------------------------------------------------------------

class CLBagDataset(Dataset):
    """Dataset for pre-computed MPNet embeddings + linguistic features.

    During training, instance dropout is applied consistently across both
    the embedding and linguistic feature tensors for each utterance.
    """

    def __init__(self, interviews: list[dict], instance_dropout: float = 0.0):
        self.interviews = interviews
        self.instance_dropout = instance_dropout

    def __len__(self) -> int:
        return len(self.interviews)

    def __getitem__(self, idx: int) -> dict:
        item = self.interviews[idx]
        patient_bag  = item["patient_embeddings"]       # (P, 768)
        patient_ling = item["patient_ling_features"]    # (P, N_FEATURES)
        intv_bag     = item["interviewer_embeddings"]   # (I, 768)

        # Consistent instance dropout on (embedding, ling) pairs
        if self.instance_dropout > 0:
            P = patient_bag.size(0)
            if P > 2:
                mask = torch.rand(P) > self.instance_dropout
                mask[0] = True
                if mask.sum() < 2:
                    mask[:2] = True
                patient_bag  = patient_bag[mask]
                patient_ling = patient_ling[mask]

            I = intv_bag.size(0)
            if I > 2:
                imask = torch.rand(I) > self.instance_dropout
                imask[0] = True
                if imask.sum() < 2:
                    imask[:2] = True
                intv_bag = intv_bag[imask]

        return {
            "patient_bag":  patient_bag,
            "patient_ling": patient_ling,
            "intv_bag":     intv_bag,
            "label":        torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
        }


def collate_cl_bags(batch: list[dict]) -> dict:
    """Pad variable-length bags into a batch tensor."""
    patient_bags  = [b["patient_bag"]  for b in batch]
    patient_lings = [b["patient_ling"] for b in batch]
    intv_bags     = [b["intv_bag"]     for b in batch]
    labels        = torch.stack([b["label"] for b in batch])
    ids           = [b["interview_id"] for b in batch]

    patient_sizes  = [x.size(0) for x in patient_bags]
    intv_sizes     = [x.size(0) for x in intv_bags]
    max_p = max(patient_sizes)
    max_i = max(intv_sizes)
    emb_d = patient_bags[0].size(1)
    lng_d = patient_lings[0].size(1)

    padded_p    = torch.zeros(len(batch), max_p, emb_d)
    padded_ling = torch.zeros(len(batch), max_p, lng_d)
    padded_i    = torch.zeros(len(batch), max_i, emb_d)

    for k, (pb, pl, ib) in enumerate(zip(patient_bags, patient_lings, intv_bags)):
        padded_p[k, :patient_sizes[k], :]    = pb
        padded_ling[k, :patient_sizes[k], :] = pl
        padded_i[k, :intv_sizes[k], :]       = ib

    return {
        "patient_bags":    padded_p,
        "patient_lings":   padded_ling,
        "interviewer_bags": padded_i,
        "patient_sizes":   patient_sizes,
        "interviewer_sizes": intv_sizes,
        "labels":          labels,
        "interview_ids":   ids,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss.  FL = -α·(1-p)^γ·log(p)"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two Bernoulli distributions."""
    pa = torch.clamp(torch.sigmoid(logits_a), 1e-7, 1 - 1e-7)
    pb = torch.clamp(torch.sigmoid(logits_b), 1e-7, 1 - 1e-7)
    kl_ab = pa * (pa.log() - pb.log()) + (1 - pa) * ((1 - pa).log() - (1 - pb).log())
    kl_ba = pb * (pb.log() - pa.log()) + (1 - pb) * ((1 - pb).log() - (1 - pa).log())
    return (kl_ab + kl_ba).mean() / 2.0


# ---------------------------------------------------------------------------
# LR schedule: cosine annealing with linear warmup
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing (no restarts)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self) -> None:
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            scale = e / max(1, self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale    = self.eta_min / self.base_lrs[0] + 0.5 * (1.0 - self.eta_min / self.base_lrs[0]) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]



# ---------------------------------------------------------------------------
# SWA collector
# ---------------------------------------------------------------------------

class SWACollector:
    """Collects model state dicts and averages them (uniform weighting)."""

    def __init__(self):
        self.states: list[dict] = []

    def update(self, model: nn.Module) -> None:
        self.states.append(copy.deepcopy(model.state_dict()))

    def apply(self, model: nn.Module) -> None:
        """Load the averaged state dict into `model`."""
        if not self.states:
            return
        avg = {}
        for key in self.states[0]:
            avg[key] = torch.stack([s[key].float() for s in self.states]).mean(0)
        model.load_state_dict(avg)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch_rdrop(
    model: DAMILCLClassifier,
    loader: DataLoader,
    focal_fn: FocalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rdrop_alpha: float = 0.3,
    max_grad_norm: float = 1.0,
    noise_std: float = 0.05,
    label_smoothing: float = 0.05,
) -> tuple[float, dict]:
    """One training epoch with R-Drop consistency regularization.

    For each batch:
      1. Run backbone → bag representation z.
      2. Apply classifier HEAD twice (different dropout masks) → logit_1, logit_2.
      3. Loss = [focal(logit_1, y) + focal(logit_2, y)] / 2
               + rdrop_alpha · sym_KL(logit_1, logit_2)
    """
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()
        target = batch["labels"].to(device)

        # Optional label smoothing
        smooth_target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing

        # ── Backbone (shared) ────────────────────────────────────────────
        bag_repr = model.forward_backbone(
            batch["patient_bags"].to(device),
            batch["patient_lings"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )  # (B, bag_dim)

        # ── Two head passes (different dropout masks) ────────────────────
        logit_1 = model.classifier(bag_repr).squeeze(-1)
        logit_2 = model.classifier(bag_repr).squeeze(-1)

        main_loss = (focal_fn(logit_1, smooth_target) + focal_fn(logit_2, smooth_target)) / 2.0
        kl_loss   = symmetric_kl(logit_1, logit_2)
        loss      = main_loss + rdrop_alpha * kl_loss

        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid((logit_1 + logit_2) / 2.0).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    metrics  = compute_metrics(
        np.array(all_labels),
        (np.array(all_probs) >= 0.5).astype(int),
        np.array(all_probs),
    )
    return avg_loss, metrics


@torch.no_grad()
def evaluate_cl(
    model: DAMILCLClassifier,
    loader: DataLoader,
    focal_fn: FocalLoss,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict, dict]:
    """Evaluate DAMIL-CL on a data loader.

    Returns:
        avg_loss, metrics dict, predictions dict.
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_ids = [], [], []

    for batch in loader:
        target = batch["labels"].to(device)
        logits, _, _ = model.forward_batch(
            batch["patient_bags"].to(device),
            batch["patient_lings"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
        )
        loss = focal_fn(logits, target)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())
        all_ids.extend(batch["interview_ids"])

    avg_loss = total_loss / max(len(loader), 1)
    y_true   = np.array(all_labels)
    y_prob   = np.array(all_probs)
    y_pred   = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    preds   = {
        "interview_id":     all_ids,
        "true_label":       [int(l) for l in all_labels],
        "probability":      [float(p) for p in all_probs],
        "predicted_label":  [int(p) for p in y_pred],
    }
    return avg_loss, metrics, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DAMIL-CL Cross-Validation")

    # Paths
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_cl_cv")

    # CV strategy
    parser.add_argument("--n_folds",  type=int,   default=5)
    parser.add_argument("--n_seeds",  type=int,   default=5,
                        help="Training seeds per fold for stability.")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of training subjects held out for validation.")

    # Encoder
    parser.add_argument("--encoder_name", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--pooling",      type=str,   default="mean",
                        choices=["mean", "cls"])
    parser.add_argument("--window_size",  type=int,   default=0,
                        help="Sliding-window context (0 = disabled).")

    # Model architecture
    parser.add_argument("--proj_dim",      type=int,   default=64)
    parser.add_argument("--ling_proj_dim", type=int,   default=16)
    parser.add_argument("--att_hidden",    type=int,   default=64)
    parser.add_argument("--dropout_rate",  type=float, default=0.35)

    # Training
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--max_epochs",        type=int,   default=80)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight_decay",      type=float, default=2e-4)
    parser.add_argument("--warmup_epochs",     type=int,   default=5)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--noise_std",         type=float, default=0.05)
    parser.add_argument("--instance_dropout",  type=float, default=0.15)
    parser.add_argument("--label_smoothing",   type=float, default=0.05)
    parser.add_argument("--focal_gamma",       type=float, default=2.0)
    parser.add_argument("--rdrop_alpha",       type=float, default=0.3)
    parser.add_argument("--swa_start_frac",    type=float, default=0.90,
                        help="Fraction of training epochs after which SWA collection starts.")
    parser.add_argument("--patience",          type=int,   default=15)
    parser.add_argument("--checkpoint_metric", type=str,   default="val_loss",
                        choices=["val_loss", "f1", "roc_auc", "balanced_accuracy"])
    parser.add_argument("--threshold_metric",  type=str,   default="f1",
                        choices=["f1", "balanced_accuracy", "loss"],
                        help="Metric to optimize when tuning the decision threshold.")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
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
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device(
        "cuda"  if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Data loading ────────────────────────────────────────────────────
    logger.info("Loading all interviews…")
    all_interviews = load_all_interviews_with_roles(args.data_dir)

    # Pre-compute frozen sentence embeddings (once, before CV)
    logger.info(f"Encoding utterances with {args.encoder_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder   = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embedding_dim = encoder.config.hidden_size

    all_interviews = precompute_dual_role_embeddings(
        all_interviews, tokenizer, encoder, device,
        max_len=args.max_len, pooling=args.pooling,
        window_size=args.window_size,
    )

    # Pre-compute raw linguistic features (once, before CV)
    # Normalization is done INSIDE train_eval_fn to prevent leakage.
    logger.info("Extracting clinical linguistic features…")
    all_interviews = extract_all_ling_features(all_interviews)

    logger.info(
        f"Dataset: {len(all_interviews)} interviews, "
        f"embedding_dim={embedding_dim}, n_ling_features={N_FEATURES}"
    )

    # ── Per-fold training / evaluation callback ──────────────────────────
    def train_eval_fn(train_pool: list[dict], test_set: list[dict], run_seed: int) -> dict:
        set_seed(run_seed)

        # ── Internal train / val split (subject-level, stratified) ──────
        subj_labels = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids   = sorted(subj_labels)
        u_labels = [subj_labels[s] for s in u_sids]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=run_seed
        )
        tr_idx, va_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))
        train_sids = {u_sids[i] for i in tr_idx}
        val_sids   = {u_sids[i] for i in va_idx}

        train_raw = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_raw   = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        # ── Leakage-free linguistic feature normalization ────────────────
        # Fit only on training utterances; transform val and test.
        train_data, ling_mean, ling_std = normalize_ling_features(train_raw, fit=True)
        val_data,  _, _  = normalize_ling_features(val_raw,  mean=ling_mean, std=ling_std)
        test_data, _, _  = normalize_ling_features(test_set, mean=ling_mean, std=ling_std)

        # ── DataLoaders ──────────────────────────────────────────────────
        train_ds = CLBagDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds   = CLBagDataset(val_data)
        test_ds  = CLBagDataset(test_data)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_cl_bags,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_cl_bags,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_cl_bags,
        )

        # ── Model ────────────────────────────────────────────────────────
        model = DAMILCLClassifier(
            embedding_dim=embedding_dim,
            ling_dim=N_FEATURES,
            proj_dim=args.proj_dim,
            ling_proj_dim=args.ling_proj_dim,
            att_hidden=args.att_hidden,
            dropout_rate=args.dropout_rate,
        ).to(device)

        logger.info(
            f"   Model params: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # ── Loss with class-frequency weighting ──────────────────────────
        n_pos = sum(1 for iv in train_data if iv["label"] == 1)
        n_neg = len(train_data) - n_pos
        focal_alpha = n_neg / max(n_pos, 1)
        focal_fn = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma).to(device)

        # ── Optimizer + LR schedule ──────────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.max_epochs,
        )

        # ── SWA ──────────────────────────────────────────────────────────
        swa = SWACollector()
        swa_start_epoch = max(1, int(args.swa_start_frac * args.max_epochs))

        # ── Early stopping bookkeeping ───────────────────────────────────
        best_score        = float("inf") if args.checkpoint_metric == "val_loss" else -float("inf")
        epochs_no_improve = 0
        ckpt_path         = out_dir / f"_ckpt_{run_seed}.pt"

        # ── Training loop ────────────────────────────────────────────────
        for epoch in range(1, args.max_epochs + 1):
            tr_loss, tr_metrics = train_epoch_rdrop(
                model, train_loader, focal_fn, optimizer, device,
                rdrop_alpha=args.rdrop_alpha,
                max_grad_norm=args.max_grad_norm,
                noise_std=args.noise_std,
                label_smoothing=args.label_smoothing,
            )
            scheduler.step()
            v_loss, v_metrics, _ = evaluate_cl(model, val_loader, focal_fn, device)

            logger.info(
                f"   [E{epoch:02d}] tr={tr_loss:.4f} val={v_loss:.4f} "
                f"F1={v_metrics['f1']:.4f} BACC={v_metrics['balanced_accuracy']:.4f} "
                f"AUC={v_metrics['roc_auc']:.4f}"
            )

            # SWA collection
            if epoch >= swa_start_epoch:
                swa.update(model)

            # Early stopping & checkpoint
            score  = v_loss if args.checkpoint_metric == "val_loss" else v_metrics[args.checkpoint_metric]
            is_best = (score < best_score) if args.checkpoint_metric == "val_loss" else (score > best_score)
            if is_best:
                best_score        = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(f"   Early stop at epoch {epoch}.")
                    break

        # ── Apply SWA weights ────────────────────────────────────────────
        # Use SWA model if enough snapshots were collected; otherwise best ckpt.
        if len(swa.states) >= 3:
            swa.apply(model)
            logger.info(f"   SWA applied ({len(swa.states)} snapshots).")
        else:
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            logger.info("   Best checkpoint loaded (SWA skipped — too few epochs).")

        ckpt_path.unlink(missing_ok=True)

        # ── Threshold tuning on validation set ───────────────────────────
        _, _, v_preds = evaluate_cl(model, val_loader, focal_fn, device)
        best_t = find_best_threshold(
            np.array(v_preds["true_label"]),
            np.array(v_preds["probability"]),
            metric=args.threshold_metric,
            pos_weight=focal_alpha,
        )
        logger.info(f"   Best threshold ({args.threshold_metric}): {best_t:.3f}")

        # ── Test evaluation ───────────────────────────────────────────────
        _, test_metrics, test_preds = evaluate_cl(
            model, test_loader, focal_fn, device, threshold=best_t
        )
        return {**test_metrics, **test_preds}

    # ── Run K-fold CV ────────────────────────────────────────────────────
    agg_metrics, raw_metrics = run_stratified_group_k_fold(
        interviews=all_interviews,
        train_eval_fn=train_eval_fn,
        n_folds=args.n_folds,
        n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    # ── Save results ─────────────────────────────────────────────────────
    def _json_default(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        raise TypeError(type(obj))

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({"aggregate": agg_metrics, "raw": raw_metrics}, f,
                  indent=4, default=_json_default)

    report = format_aggregate_report(agg_metrics)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write("# DAMIL-CL K-Fold CV Results\n\n")
        f.write(report)

    logger.info("\n" + report)
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
