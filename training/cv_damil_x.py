"""DAMIL-X Stratified Group K-Fold Cross-Validation.

Target: surpass SS-DAMIL-R v9d (ROC-AUC 0.8245 / F1 0.6290 / BAcc 0.7407 / Recall 0.7645).

Pipeline (strictly leakage-free):
    Q-A dialogue pairs → frozen MPNet → DAMIL-X (multi-view MIL + aux PHQ-8 sum)
    → Focal + R-Drop + Manifold Mixup + SWA + MC-dropout inference → threshold
    tuned on val → test metrics aggregated over 5 folds × N seeds.
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

from dataset import load_all_interviews_dialogue_pairs
from extract_linguistic_features import (
    N_FEATURES,
    extract_all_ling_features,
    normalize_ling_features,
)
from models.damil_x import DAMILXClassifier
from training.train_damil_r import (
    precompute_dialogue_pair_embeddings,
    set_seed,
)
from utils.evaluation import run_stratified_group_k_fold
from utils.metrics import compute_metrics, find_best_threshold
from utils.stats import format_aggregate_report


# ---------------------------------------------------------------------------
# Dataset / collation
# ---------------------------------------------------------------------------

class DAMILXDataset(Dataset):
    def __init__(self, interviews: list[dict], instance_dropout: float = 0.0):
        self.interviews = interviews
        self.instance_dropout = instance_dropout

    def __len__(self) -> int:
        return len(self.interviews)

    def __getitem__(self, idx: int) -> dict:
        it = self.interviews[idx]
        p_bag = it["patient_embeddings"]
        p_ling = it["patient_ling_features"]
        i_bag = it["interviewer_embeddings"]

        if self.instance_dropout > 0:
            P = p_bag.size(0)
            if P > 2:
                mask = torch.rand(P) > self.instance_dropout
                mask[0] = True
                if mask.sum() < 2:
                    mask[:2] = True
                p_bag = p_bag[mask]
                p_ling = p_ling[mask]

            I = i_bag.size(0)
            if I > 2:
                im = torch.rand(I) > self.instance_dropout
                im[0] = True
                if im.sum() < 2:
                    im[:2] = True
                i_bag = i_bag[im]

        return {
            "patient_bag": p_bag,
            "patient_ling": p_ling,
            "intv_bag": i_bag,
            "label": torch.tensor(it["label"], dtype=torch.float),
            "aux_target": torch.tensor(it["aux_target"], dtype=torch.float),
            "has_aux": torch.tensor(it["has_aux"], dtype=torch.float),
            "interview_id": it["interview_id"],
        }


def collate_damil_x(batch: list[dict]) -> dict:
    p_bags = [b["patient_bag"] for b in batch]
    p_lings = [b["patient_ling"] for b in batch]
    i_bags = [b["intv_bag"] for b in batch]

    p_sizes = [x.size(0) for x in p_bags]
    i_sizes = [x.size(0) for x in i_bags]
    max_p = max(p_sizes)
    max_i = max(i_sizes)
    emb_d = p_bags[0].size(1)
    lng_d = p_lings[0].size(1)

    padded_p = torch.zeros(len(batch), max_p, emb_d)
    padded_l = torch.zeros(len(batch), max_p, lng_d)
    padded_i = torch.zeros(len(batch), max_i, emb_d)

    for k, (pb, pl, ib) in enumerate(zip(p_bags, p_lings, i_bags)):
        padded_p[k, :p_sizes[k], :] = pb
        padded_l[k, :p_sizes[k], :] = pl
        padded_i[k, :i_sizes[k], :] = ib

    return {
        "patient_bags": padded_p,
        "patient_lings": padded_l,
        "interviewer_bags": padded_i,
        "patient_sizes": p_sizes,
        "interviewer_sizes": i_sizes,
        "labels": torch.stack([b["label"] for b in batch]),
        "aux_target": torch.stack([b["aux_target"] for b in batch]),
        "has_aux": torch.stack([b["has_aux"] for b in batch]),
        "interview_ids": [b["interview_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def symmetric_kl_bernoulli(la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    pa = torch.clamp(torch.sigmoid(la), 1e-7, 1 - 1e-7)
    pb = torch.clamp(torch.sigmoid(lb), 1e-7, 1 - 1e-7)
    kl_ab = pa * (pa.log() - pb.log()) + (1 - pa) * ((1 - pa).log() - (1 - pb).log())
    kl_ba = pb * (pb.log() - pa.log()) + (1 - pb) * ((1 - pb).log() - (1 - pa).log())
    return (kl_ab + kl_ba).mean() / 2.0


# ---------------------------------------------------------------------------
# LR schedule + SWA
# ---------------------------------------------------------------------------

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.opt = optimizer
        self.we = warmup_epochs
        self.te = total_epochs
        self.emin = eta_min
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.e = 0

    def step(self):
        self.e += 1
        if self.e <= self.we:
            s = self.e / max(1, self.we)
        else:
            p = (self.e - self.we) / max(1, self.te - self.we)
            frac = self.emin / self.base[0]
            s = frac + 0.5 * (1 - frac) * (1 + math.cos(math.pi * p))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * s

    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]


class SWACollector:
    def __init__(self):
        self.states = []

    def update(self, model):
        self.states.append(copy.deepcopy(model.state_dict()))

    def apply(self, model):
        if not self.states:
            return
        avg = {}
        for k in self.states[0]:
            avg[k] = torch.stack([s[k].float() for s in self.states]).mean(0)
        model.load_state_dict(avg)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, focal_fn, optimizer, device,
    aux_weight: float,
    rdrop_alpha: float,
    mixup_alpha: float,
    noise_std: float,
    label_smoothing: float,
    max_grad_norm: float,
    aux_scale: float,
):
    model.train()
    total = 0.0

    for batch in loader:
        optimizer.zero_grad()
        y = batch["labels"].to(device)
        aux_y = batch["aux_target"].to(device)
        mask = batch["has_aux"].to(device)

        ys = y * (1 - label_smoothing) + 0.5 * label_smoothing

        bag = model.forward_backbone(
            batch["patient_bags"].to(device),
            batch["patient_lings"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )

        # Manifold Mixup on bag repr
        if mixup_alpha > 0 and bag.size(0) >= 2:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(bag.size(0), device=device)
            bag_mix = lam * bag + (1 - lam) * bag[perm]
            y_mix = lam * ys + (1 - lam) * ys[perm]
            aux_mix = lam * aux_y + (1 - lam) * aux_y[perm]
            mask_mix = torch.maximum(mask, mask[perm])
            mix_out = model.forward_heads(bag_mix)
            loss_mix = focal_fn(mix_out["logits"], y_mix) + aux_weight * masked_mse(
                mix_out["aux"] * aux_scale, aux_mix, mask_mix,
            )
        else:
            loss_mix = torch.tensor(0.0, device=device)

        # R-Drop: two head passes on same bag repr (different dropout masks)
        out1 = model.forward_heads(bag)
        out2 = model.forward_heads(bag)

        l1 = focal_fn(out1["logits"], ys) + aux_weight * masked_mse(
            out1["aux"] * aux_scale, aux_y, mask,
        )
        l2 = focal_fn(out2["logits"], ys) + aux_weight * masked_mse(
            out2["aux"] * aux_scale, aux_y, mask,
        )
        kl = symmetric_kl_bernoulli(out1["logits"], out2["logits"])

        loss = 0.5 * (l1 + l2) + rdrop_alpha * kl + loss_mix
        if not torch.isfinite(loss):
            continue

        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total += loss.item()

    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, focal_fn, device, threshold: float = 0.5, mc_passes: int = 1):
    model.eval()
    if mc_passes > 1:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    total = 0.0
    all_probs, all_y, all_ids = [], [], []

    for batch in loader:
        y = batch["labels"].to(device)
        bag = model.forward_backbone(
            batch["patient_bags"].to(device),
            batch["patient_lings"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
        )
        if mc_passes > 1:
            probs_sum = 0.0
            last_logits = None
            for _ in range(mc_passes):
                out = model.forward_heads(bag)
                probs_sum = probs_sum + torch.sigmoid(out["logits"])
                last_logits = out["logits"]
            probs = (probs_sum / mc_passes).cpu().numpy()
            loss = focal_fn(last_logits, y)
        else:
            out = model.forward_heads(bag)
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            loss = focal_fn(out["logits"], y)

        total += loss.item()
        all_probs.extend(probs)
        all_y.extend(y.cpu().numpy())
        all_ids.extend(batch["interview_ids"])

    avg = total / max(1, len(loader))
    y_true = np.array(all_y)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    return (
        avg,
        compute_metrics(y_true, y_pred, y_prob),
        {
            "interview_id": all_ids,
            "true_label": [int(v) for v in all_y],
            "probability": [float(v) for v in all_probs],
            "predicted_label": [int(v) for v in y_pred],
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DAMIL-X Cross-Validation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/damil_x_cv")

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)

    parser.add_argument("--encoder_name", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Token length cap for Q-A dialogue pairs.")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls"])

    parser.add_argument("--proj_dim", type=int, default=80)
    parser.add_argument("--ling_proj_dim", type=int, default=16)
    parser.add_argument("--att_hidden", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.35)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--instance_dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rdrop_alpha", type=float, default=0.3)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)

    parser.add_argument("--aux_weight_start", type=float, default=0.3)
    parser.add_argument("--aux_weight_end", type=float, default=0.05)

    parser.add_argument("--swa_start_frac", type=float, default=0.80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mc_passes", type=int, default=10)
    parser.add_argument("--threshold_metric", type=str, default="f1",
                        choices=["f1", "balanced_accuracy", "loss"])

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "cv_run.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load dialogue-pair interviews (Q-A + interviewer + per-symptom) ──
    all_interviews = load_all_interviews_dialogue_pairs(args.data_dir)

    # Align utterances used by linguistic feature extractor with Q-A instances.
    # `utterances` from process_transcript is the per-Participant merged-turn
    # list, which is 1-1 with `qa_pairs`. Ling features indexed by turn idx.
    all_interviews = extract_all_ling_features(all_interviews)

    # Precompute encoder embeddings for Q-A pairs (patient) + Ellie utterances.
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embedding_dim = encoder.config.hidden_size

    logger.info(f"Encoding Q-A pairs with {args.encoder_name}")
    all_interviews = precompute_dialogue_pair_embeddings(
        all_interviews, tokenizer, encoder, device,
        max_len=args.max_len, pooling=args.pooling,
    )

    # Compute aux target = PHQ-8 sum / 24 (∈ [0, 1]) + has_aux mask
    for iv in all_interviews:
        if iv.get("has_symptoms", False):
            iv["aux_target"] = float(sum(iv["symptoms"])) / 24.0
            iv["has_aux"] = 1.0
        else:
            iv["aux_target"] = 0.0
            iv["has_aux"] = 0.0

    # Sanity: Q-A count must match ling feature count
    for iv in all_interviews:
        n_qa = iv["patient_embeddings"].size(0)
        n_lg = iv["patient_ling_features"].size(0)
        if n_qa != n_lg:
            # Trim to shorter to preserve 1-1 alignment (edge case protection).
            m = min(n_qa, n_lg)
            iv["patient_embeddings"] = iv["patient_embeddings"][:m]
            iv["patient_ling_features"] = iv["patient_ling_features"][:m]

    logger.info(f"Total interviews: {len(all_interviews)}")

    # ── Per-fold training callback ───────────────────────────────────────
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        subj_labels = {iv["interview_id"]: iv["label"] for iv in train_pool}
        u_sids = sorted(subj_labels)
        u_labels = [subj_labels[s] for s in u_sids]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=run_seed,
        )
        tr_idx, va_idx = next(sss.split(np.zeros(len(u_labels)), u_labels))
        train_sids = {u_sids[i] for i in tr_idx}
        val_sids = {u_sids[i] for i in va_idx}
        test_sids = {iv["interview_id"] for iv in test_set}

        # Leakage assertions
        assert train_sids.isdisjoint(test_sids), "leak: train ∩ test"
        assert val_sids.isdisjoint(test_sids), "leak: val ∩ test"
        assert train_sids.isdisjoint(val_sids), "leak: train ∩ val"

        train_raw = [iv for iv in train_pool if iv["interview_id"] in train_sids]
        val_raw = [iv for iv in train_pool if iv["interview_id"] in val_sids]

        # Leakage-free linguistic feature z-score: fit on train only
        train_data, lm, ls = normalize_ling_features(train_raw, fit=True)
        val_data, _, _ = normalize_ling_features(val_raw, mean=lm, std=ls)
        test_data, _, _ = normalize_ling_features(test_set, mean=lm, std=ls)

        train_ds = DAMILXDataset(train_data, instance_dropout=args.instance_dropout)
        val_ds = DAMILXDataset(val_data)
        test_ds = DAMILXDataset(test_data)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_damil_x,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_damil_x,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_damil_x,
        )

        model = DAMILXClassifier(
            embedding_dim=embedding_dim,
            ling_dim=N_FEATURES,
            proj_dim=args.proj_dim,
            ling_proj_dim=args.ling_proj_dim,
            att_hidden=args.att_hidden,
            dropout_rate=args.dropout_rate,
        ).to(device)

        n_pos = sum(1 for iv in train_data if iv["label"] == 1)
        n_neg = len(train_data) - n_pos
        focal_alpha = n_neg / max(1, n_pos)
        focal_fn = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosine(
            optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.max_epochs,
        )

        swa = SWACollector()
        swa_start = max(1, int(args.swa_start_frac * args.max_epochs))

        best_val = float("inf")
        patience_ct = 0
        ckpt = out_dir / f"_ckpt_{run_seed}.pt"

        for epoch in range(1, args.max_epochs + 1):
            # Aux weight linear anneal
            frac = min(1.0, (epoch - 1) / max(1, args.max_epochs - 1))
            aux_w = args.aux_weight_start + frac * (args.aux_weight_end - args.aux_weight_start)

            tr_loss = train_epoch(
                model, train_loader, focal_fn, optimizer, device,
                aux_weight=aux_w,
                rdrop_alpha=args.rdrop_alpha,
                mixup_alpha=args.mixup_alpha,
                noise_std=args.noise_std,
                label_smoothing=args.label_smoothing,
                max_grad_norm=args.max_grad_norm,
                aux_scale=1.0,
            )
            scheduler.step()
            v_loss, v_metrics, _ = evaluate(model, val_loader, focal_fn, device)

            if epoch == 1 or epoch % 5 == 0:
                logger.info(
                    f"   [E{epoch:02d}] tr={tr_loss:.4f} val={v_loss:.4f} "
                    f"F1={v_metrics['f1']:.4f} BACC={v_metrics['balanced_accuracy']:.4f} "
                    f"AUC={v_metrics['roc_auc']:.4f} auxW={aux_w:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if epoch >= swa_start:
                swa.update(model)

            if v_loss < best_val:
                best_val = v_loss
                patience_ct = 0
                torch.save(model.state_dict(), ckpt)
            else:
                patience_ct += 1
                if patience_ct >= args.patience:
                    logger.info(f"   Early stop @ E{epoch}")
                    break

        # Choose best model: SWA if ≥3 snapshots AND validates better, else best ckpt
        if ckpt.exists():
            best_sd = torch.load(ckpt, weights_only=True)
        else:
            best_sd = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_sd)
        _, bm, _ = evaluate(model, val_loader, focal_fn, device, mc_passes=1)
        best_auc = bm.get("roc_auc", 0.0)

        if len(swa.states) >= 3:
            swa_model = DAMILXClassifier(
                embedding_dim=embedding_dim, ling_dim=N_FEATURES,
                proj_dim=args.proj_dim, ling_proj_dim=args.ling_proj_dim,
                att_hidden=args.att_hidden, dropout_rate=args.dropout_rate,
            ).to(device)
            swa.apply(swa_model)
            _, sm, _ = evaluate(swa_model, val_loader, focal_fn, device, mc_passes=1)
            swa_auc = sm.get("roc_auc", 0.0)
            if swa_auc >= best_auc:
                logger.info(f"   SWA preferred: AUC {best_auc:.4f} → {swa_auc:.4f}")
                model = swa_model

        # Threshold tuning (MC-dropout probs)
        _, _, v_preds = evaluate(model, val_loader, focal_fn, device, mc_passes=args.mc_passes)
        best_t = find_best_threshold(
            np.array(v_preds["true_label"]),
            np.array(v_preds["probability"]),
            metric=args.threshold_metric,
            pos_weight=focal_alpha,
        )

        _, test_metrics, test_preds = evaluate(
            model, test_loader, focal_fn, device,
            threshold=best_t, mc_passes=args.mc_passes,
        )

        ckpt.unlink(missing_ok=True)
        return {**test_metrics, **test_preds}

    agg, raw = run_stratified_group_k_fold(
        interviews=all_interviews,
        train_eval_fn=train_eval_fn,
        n_folds=args.n_folds,
        n_seeds_per_fold=args.n_seeds,
        random_state=args.seed,
    )

    def _json_default(o):
        if isinstance(o, (np.ndarray, np.generic)):
            return o.tolist() if isinstance(o, np.ndarray) else o.item()
        raise TypeError(type(o))

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({"aggregate": agg, "raw": raw}, f, indent=4, default=_json_default)

    report = format_aggregate_report(agg)
    with open(out_dir / "cv_report.txt", "w") as f:
        f.write("# DAMIL-X K-Fold CV Results\n\n")
        f.write(report)
    logger.info("\n" + report)
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
