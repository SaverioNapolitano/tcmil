"""TC-MIL on the official AVEC2017 DAIC-WOZ protocol.

Protocol (matches the text-only literature):
    - Train on the official train split (107 subjects).
    - Model selection (early stopping) and decision threshold on dev (35).
    - Seed-ensemble: average predicted probabilities over N seeds.
    - Test (47) is evaluated once, with the dev-tuned threshold, only when
      --eval_test is passed. Dev ablations never touch it.

No subject appears in more than one split (asserted at load time), the
encoder is frozen (so pre-computing embeddings cannot leak), and no
statistic of dev/test is used during training other than dev early stopping.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.models.tcmil import TCMIL
from src.core.tcmil_data import assert_no_leakage, embed_chunks, load_official_split
from src.core.utils.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChunkBagDataset(Dataset):
    def __init__(self, interviews, instance_dropout: float = 0.0):
        self.interviews = interviews
        self.instance_dropout = instance_dropout

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        iv = self.interviews[idx]
        bag = iv["embeddings"]
        if self.instance_dropout > 0 and bag.size(0) > 4:
            keep = torch.rand(bag.size(0)) > self.instance_dropout
            if keep.sum() < 4:
                keep[:4] = True
            bag = bag[keep]
        sym = torch.tensor(iv["symptoms"], dtype=torch.float)
        return {
            "bag": bag,
            "label": torch.tensor(iv["label"], dtype=torch.float),
            "symptoms_bin": (sym >= 1).float(),
            "symptoms_raw": sym / 3.0,
            "has_symptoms": torch.tensor(float(iv["has_symptoms"])),
        }


def collate(batch):
    bags = [b["bag"] for b in batch]
    n_max = max(b.size(0) for b in bags)
    d = bags[0].size(1)
    padded = torch.zeros(len(bags), n_max, d)
    mask = torch.zeros(len(bags), n_max)
    for i, b in enumerate(bags):
        padded[i, : b.size(0)] = b
        mask[i, : b.size(0)] = 1
    return {
        "bags": padded,
        "mask": mask,
        "labels": torch.stack([b["label"] for b in batch]),
        "symptoms_bin": torch.stack([b["symptoms_bin"] for b in batch]),
        "symptoms_raw": torch.stack([b["symptoms_raw"] for b in batch]),
        "has_symptoms": torch.stack([b["has_symptoms"] for b in batch]),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        out = model(batch["bags"].to(device), batch["mask"].to(device))
        probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().tolist())
        labels.extend(batch["labels"].numpy().tolist())
    return np.array(probs), np.array(labels)


@torch.no_grad()
def predict_symptom_sum(model, loader, device):
    """Per-bag summed PHQ-8 symptom probability (the aux head as a clinical
    prior; lever 4). Returns array aligned with predict()."""
    model.eval()
    out_sum = []
    for batch in loader:
        out = model(batch["bags"].to(device), batch["mask"].to(device))
        out_sum.extend(torch.sigmoid(out["symptom_logits"]).sum(-1).cpu().numpy().tolist())
    return np.array(out_sum)


def train_one_seed(args, seed, train_ivs, dev_ivs, device, embedding_dim):
    set_seed(seed)

    train_loader = DataLoader(
        ChunkBagDataset(train_ivs, instance_dropout=args.instance_dropout),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    dev_loader = DataLoader(
        ChunkBagDataset(dev_ivs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
    )

    model = TCMIL(
        embedding_dim=embedding_dim, proj_dim=args.proj_dim,
        attn_dim=args.attn_dim, dropout=args.dropout,
        temporal=getattr(args, "temporal", "none"),
        gru_layers=getattr(args, "gru_layers", 1),
    ).to(device)

    n_pos = sum(iv["label"] for iv in train_ivs)
    auto_pw = (len(train_ivs) - n_pos) / max(1, n_pos)
    pw = getattr(args, "pos_weight", -1.0)
    pos_weight = torch.tensor([auto_pw if pw < 0 else pw]).to(device)
    main_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sym_loss = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            bags = batch["bags"].to(device)
            if args.noise_std > 0:
                bags = bags + torch.randn_like(bags) * args.noise_std
            out = model(bags, batch["mask"].to(device))
            loss = main_loss(out["logits"], batch["labels"].to(device))
            if args.aux_weight > 0:
                if getattr(args, "aux_mode", "binary") == "score":
                    per_item = (torch.sigmoid(out["symptom_logits"])
                                - batch["symptoms_raw"].to(device)) ** 2
                else:
                    per_item = sym_loss(out["symptom_logits"], batch["symptoms_bin"].to(device))
                has = batch["has_symptoms"].to(device).unsqueeze(1)
                denom = has.sum() * 8
                if denom > 0:
                    loss = loss + args.aux_weight * (per_item * has).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        dev_probs, dev_labels = predict(model, dev_loader, device)
        auc = compute_metrics(dev_labels, (dev_probs >= 0.5).astype(int), dev_probs)["roc_auc"]
        if auc > best_auc:
            best_auc, no_improve = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, best_auc


def tune_threshold(y_true, probs, metric: str = "f1", prevalence: float | None = None):
    """Pick a decision threshold on a held-out (dev) set.

    Strategies:
        f1          - maximize F1 (recall-biased on tiny dev sets).
        bacc        - maximize balanced accuracy (recall/specificity trade).
        youden      - maximize Youden's J = TPR - FPR (lands near prevalence,
                      threshold-stable and N-robust).
        prevalence  - choose t so the predicted positive rate matches the
                      training prevalence; ignores noisy dev F1 entirely.
    """
    y_true = np.asarray(y_true)
    if metric == "prevalence":
        assert prevalence is not None
        # Highest threshold whose predicted-positive rate is >= prevalence.
        ts = np.arange(0.05, 0.95, 0.01)
        best_t, best_gap = 0.5, 1e9
        for t in ts:
            rate = float((probs >= t).mean())
            gap = abs(rate - prevalence)
            if gap < best_gap:
                best_gap, best_t = gap, float(t)
        return best_t

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (probs >= t).astype(int)
        m = compute_metrics(y_true, pred, probs)
        if metric == "f1":
            score = m["f1"]
        elif metric == "bacc":
            score = m["balanced_accuracy"]
        elif metric == "youden":
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            tpr = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            score = tpr - fpr
        else:
            raise ValueError(metric)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def main():
    p = argparse.ArgumentParser(description="TC-MIL official-protocol training")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/tcmil_official")
    p.add_argument("--encoder_name", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--doc_prefix", default="",
                   help='Prefix prepended to every chunk before encoding (e5: "query: ").')
    p.add_argument("--temporal", default="none", choices=["none", "gru", "transformer"],
                   help="Optional context layer over the chunk sequence before MIL pooling.")
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--pos_weight", type=float, default=-1.0,
                   help="Override BCE pos_weight; <0 = auto (neg/pos). Lower = less recall-biased.")
    p.add_argument("--pooling", default="mean", choices=["mean", "last"],
                   help="Token pooling for the encoder (last = causal-LM embedders).")
    p.add_argument("--aux_mode", default="binary", choices=["binary", "score"],
                   help="Aux target: binarized PHQ-8 items (BCE) or 0-3 subscores (MSE).")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--participant_only", action="store_true",
                   help="Drop interviewer lines from chunks (Burdisso 2024 bias control).")
    p.add_argument("--gap_merge", type=float, default=None,
                   help="Silence-gap (s) participant-only segmentation matching the "
                        "E-DAIC ASR pipeline; implies participant_only. Use the same "
                        "value for DAIC train and E-DAIC zero-shot test.")
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--instance_dropout", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--aux_weight", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--eval_test", action="store_true",
                   help="Evaluate the official test split (use only for the final config).")
    p.add_argument("--threshold_metric", default="f1",
                   choices=["f1", "bacc", "youden", "prevalence"],
                   help="Dev threshold-selection strategy for the headline number.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # --- Data (official splits, subject-disjoint by construction) ---
    train_ivs = load_official_split(args.data_dir, "train", args.window, args.stride,
                                    participant_only=args.participant_only,
                                    gap_merge=args.gap_merge)
    dev_ivs = load_official_split(args.data_dir, "dev", args.window, args.stride,
                                  participant_only=args.participant_only,
                                  gap_merge=args.gap_merge)
    test_ivs = load_official_split(args.data_dir, "test", args.window, args.stride,
                                   participant_only=args.participant_only,
                                   gap_merge=args.gap_merge)
    assert_no_leakage(train_ivs, dev_ivs, test_ivs)
    log.info(f"train={len(train_ivs)} dev={len(dev_ivs)} test={len(test_ivs)}")
    sizes = [len(iv["chunks"]) for iv in train_ivs]
    log.info(f"chunks/interview: mean={np.mean(sizes):.1f} min={min(sizes)} max={max(sizes)}")

    cache_tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
    if args.participant_only:
        cache_tag += "_ponly"
    if args.doc_prefix:
        cache_tag += "_pfx" + "".join(c for c in args.doc_prefix if c.isalnum())
    if args.pooling != "mean":
        cache_tag += f"_pool{args.pooling}"
    for ivs in (train_ivs, dev_ivs, test_ivs):
        embed_chunks(ivs, args.encoder_name, device, max_len=args.max_len,
                     cache_tag=cache_tag, prefix=args.doc_prefix, pooling=args.pooling)
    embedding_dim = train_ivs[0]["embeddings"].size(1)
    log.info(f"embedding_dim={embedding_dim}")

    dev_loader = DataLoader(ChunkBagDataset(dev_ivs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ChunkBagDataset(test_ivs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)

    # --- Multi-seed training ---
    dev_prob_runs, test_prob_runs, per_seed = [], [], []
    dev_sym_runs, test_sym_runs = [], []
    dev_labels = test_labels = None
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    # Model-construction params so a checkpoint can be rebuilt without guessing.
    json.dump({"embedding_dim": embedding_dim, "proj_dim": args.proj_dim,
               "attn_dim": args.attn_dim, "temporal": args.temporal,
               "gru_layers": args.gru_layers, "dropout": args.dropout,
               "encoder_name": args.encoder_name, "window": args.window,
               "stride": args.stride, "base_seed": args.base_seed,
               "n_seeds": args.n_seeds},
              open(ckpt_dir / "config.json", "w"), indent=2)
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        model, best_auc = train_one_seed(args, seed, train_ivs, dev_ivs, device, embedding_dim)

        # Save the trained head per seed (encoder is frozen / not in `model`, so
        # this is small) -> enables re-inference, MC-dropout, post-hoc analysis
        # without retraining. Save everything by default.
        torch.save(model.state_dict(), ckpt_dir / f"seed_{seed}.pt")

        dev_probs, dev_labels = predict(model, dev_loader, device)
        dev_prob_runs.append(dev_probs)
        dev_sym_runs.append(predict_symptom_sum(model, dev_loader, device))
        m = compute_metrics(dev_labels, (dev_probs >= 0.5).astype(int), dev_probs)
        per_seed.append(m)
        log.info(f"seed {seed}: dev AUC={m['roc_auc']:.4f} F1@0.5={m['f1']:.4f}")

        if args.eval_test:
            test_probs, test_labels = predict(model, test_loader, device)
            test_prob_runs.append(test_probs)
            test_sym_runs.append(predict_symptom_sum(model, test_loader, device))

    # --- Seed-ensemble on dev ---
    dev_avg = np.mean(dev_prob_runs, axis=0)
    train_prev = float(np.mean([iv["label"] for iv in train_ivs]))

    # Compare threshold-selection strategies (all tuned on dev only).
    strategies = {
        "f1": tune_threshold(dev_labels, dev_avg, "f1"),
        "bacc": tune_threshold(dev_labels, dev_avg, "bacc"),
        "youden": tune_threshold(dev_labels, dev_avg, "youden"),
        "prevalence": tune_threshold(dev_labels, dev_avg, "prevalence", prevalence=train_prev),
    }
    best_t = strategies[args.threshold_metric]
    dev_metrics = compute_metrics(dev_labels, (dev_avg >= best_t).astype(int), dev_avg)
    log.info(f"DEV ensemble [{args.threshold_metric}] (t={best_t:.2f}): " +
             " ".join(f"{k}={v:.4f}" for k, v in dev_metrics.items() if isinstance(v, float)))

    seed_aucs = [m["roc_auc"] for m in per_seed]
    log.info(f"per-seed dev AUC: {np.mean(seed_aucs):.4f} ± {np.std(seed_aucs):.4f}")

    results = {
        "args": vars(args),
        "per_seed_dev": per_seed,
        "dev_ensemble": dev_metrics,
        "dev_threshold": best_t,
        "thresholds": strategies,
        "train_prevalence": train_prev,
        "dev_probs": dev_avg.tolist(),
        "dev_labels": np.asarray(dev_labels).tolist(),
        "dev_prob_runs": [p.tolist() for p in dev_prob_runs],
        "dev_sym_runs": [s.tolist() for s in dev_sym_runs],
    }

    # --- Final test evaluation (dev-tuned threshold, ensemble probs) ---
    if args.eval_test:
        test_avg = np.mean(test_prob_runs, axis=0)
        results["test_probs"] = test_avg.tolist()
        results["test_labels"] = np.asarray(test_labels).tolist()
        results["test_prob_runs"] = [p.tolist() for p in test_prob_runs]
        results["test_sym_runs"] = [s.tolist() for s in test_sym_runs]
        # Literature-comparable protocol (Burdisso 2023): mean +- std of
        # PER-SEED test metrics (no seed ensembling), dev-selected threshold.
        per_seed_test = [
            compute_metrics(test_labels, (p >= best_t).astype(int), p)
            for p in test_prob_runs
        ]
        results["test_per_seed"] = per_seed_test
        for k in ("precision", "recall", "f1", "macro_f1", "micro_f1", "roc_auc"):
            vals = [m[k] for m in per_seed_test]
            log.info(f"TEST per-seed {k} (t={best_t:.2f}): "
                     f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
        # Report every strategy's test metrics so the trade-off is explicit;
        # the headline uses --threshold_metric (chosen on dev).
        results["test_by_strategy"] = {}
        for name, t in strategies.items():
            tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
            results["test_by_strategy"][name] = {"threshold": t, **tm}
            log.info(f"TEST [{name}] (t={t:.2f}): "
                     f"F1={tm['f1']:.4f} P={tm['precision']:.4f} R={tm['recall']:.4f} "
                     f"BAcc={tm['balanced_accuracy']:.4f} AUC={tm['roc_auc']:.4f}")
        results["test_ensemble"] = results["test_by_strategy"][args.threshold_metric]

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
