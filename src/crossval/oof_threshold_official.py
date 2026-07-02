"""Official-protocol TC-MIL with an out-of-fold (OOF) threshold probe.

The 33-subject dev set is too small to estimate a decision threshold: F1 /
balanced-accuracy / Youden all collapse to an extreme recall-1.0 point. This
script estimates the threshold on a much larger, leakage-free signal instead:

    1. Pool = official train + dev (142 subjects). Test is held out.
    2. Stratified Group K-Fold over the pool. For each fold, train on the
       other folds (with an inner val split for early stopping) and predict
       the held-out fold -> out-of-fold probabilities. Every pool subject
       thus receives a prediction from a model that never trained on it.
    3. Estimate the threshold on the 142 OOF probabilities (stable N).
    4. Train the final seed-ensemble on the official train split (dev used
       only for early stopping), predict test, and apply the OOF threshold.

Test is evaluated exactly once. No test statistic touches training or the
threshold.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.tcmil_data import assert_no_leakage, embed_chunks, load_official_split
from src.training.train_tcmil_official import (
    ChunkBagDataset,
    collate,
    predict,
    set_seed,
    train_one_seed,
    tune_threshold,
)
from src.core.utils.metrics import compute_metrics


def build_oof_probs(args, pool, device, embedding_dim, log):
    """K-fold OOF probabilities over the pool (one prob per subject)."""
    labels = [iv["label"] for iv in pool]
    groups = [iv["interview_id"] for iv in pool]
    skf = StratifiedGroupKFold(n_splits=args.oof_folds, shuffle=True, random_state=args.seed)
    pool_np = np.array(pool)

    oof_prob = {}  # interview_id -> averaged prob across oof seeds
    oof_label = {}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(pool_np, labels, groups), 1):
        fold_train = pool_np[tr_idx].tolist()
        fold_held = pool_np[te_idx].tolist()
        held_ids = {iv["interview_id"] for iv in fold_held}

        # Inner val split (fold-deterministic) for early stopping.
        sids = sorted(iv["interview_id"] for iv in fold_train)
        sid_lab = {iv["interview_id"]: iv["label"] for iv in fold_train}
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed + fold)
        tr_i, va_i = next(sss.split(np.zeros(len(sids)), [sid_lab[s] for s in sids]))
        va_ids = {sids[i] for i in va_i}
        inner_train = [iv for iv in fold_train if iv["interview_id"] not in va_ids]
        inner_val = [iv for iv in fold_train if iv["interview_id"] in va_ids]

        assert held_ids.isdisjoint({iv["interview_id"] for iv in fold_train}), "OOF leakage"

        held_loader = DataLoader(ChunkBagDataset(fold_held), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate)

        fold_probs = []
        for s in range(args.oof_seeds):
            seed = args.seed + fold * 100 + s
            set_seed(seed)
            model, _ = train_one_seed(args, seed, inner_train, inner_val, device, embedding_dim)
            probs, _ = predict(model, held_loader, device)
            fold_probs.append(probs)

        avg = np.mean(fold_probs, axis=0)
        for iv, pr in zip(fold_held, avg):
            oof_prob[iv["interview_id"]] = float(pr)
            oof_label[iv["interview_id"]] = iv["label"]
        log.info(f"  OOF fold {fold}/{args.oof_folds}: {len(fold_held)} subjects")

    ids = sorted(oof_prob.keys())
    return (np.array([oof_prob[i] for i in ids]),
            np.array([oof_label[i] for i in ids]))


def main():
    p = argparse.ArgumentParser(description="TC-MIL official protocol with OOF threshold probe")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/tcmil_oof")
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--doc_prefix", default="",
                   help='Prefix prepended to every chunk before encoding (e5: "query: ").')
    p.add_argument("--temporal", default="none", choices=["none", "gru", "transformer"],
                   help="Optional context layer over the chunk sequence before MIL pooling.")
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--participant_only", action="store_true",
                   help="Drop interviewer lines from chunks (Burdisso 2024 bias control).")
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
    p.add_argument("--n_seeds", type=int, default=10)        # final ensemble
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=-1.0,
                   help="BCE pos_weight; <0 = auto (neg/pos). 1.0 = no weighting.")
    p.add_argument("--seed", type=int, default=42)           # OOF CV seed
    p.add_argument("--oof_folds", type=int, default=5)
    p.add_argument("--oof_seeds", type=int, default=3)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--threshold_metric", default="prevalence",
                   choices=["f1", "bacc", "youden", "prevalence"])
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

    train_ivs = load_official_split(args.data_dir, "train", args.window, args.stride,
                                    participant_only=args.participant_only)
    dev_ivs = load_official_split(args.data_dir, "dev", args.window, args.stride,
                                  participant_only=args.participant_only)
    test_ivs = load_official_split(args.data_dir, "test", args.window, args.stride,
                                   participant_only=args.participant_only)
    assert_no_leakage(train_ivs, dev_ivs, test_ivs)

    cache_tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
    if args.participant_only:
        cache_tag += "_ponly"
    if args.doc_prefix:
        cache_tag += "_pfx" + "".join(c for c in args.doc_prefix if c.isalnum())
    for ivs in (train_ivs, dev_ivs, test_ivs):
        embed_chunks(ivs, args.encoder_name, device, max_len=args.max_len,
                     cache_tag=cache_tag, prefix=args.doc_prefix)
    embedding_dim = train_ivs[0]["embeddings"].size(1)
    pool = train_ivs + dev_ivs
    pool_prev = float(np.mean([iv["label"] for iv in pool]))

    # --- 1-3: OOF probe over the pool ---
    log.info("Building OOF probabilities over train+dev pool...")
    oof_probs, oof_labels = build_oof_probs(args, pool, device, embedding_dim, log)
    oof_auc = compute_metrics(oof_labels, (oof_probs >= 0.5).astype(int), oof_probs)["roc_auc"]
    log.info(f"OOF: N={len(oof_labels)} AUC={oof_auc:.4f} prevalence={pool_prev:.3f}")

    oof_thresholds = {
        "f1": tune_threshold(oof_labels, oof_probs, "f1"),
        "bacc": tune_threshold(oof_labels, oof_probs, "bacc"),
        "youden": tune_threshold(oof_labels, oof_probs, "youden"),
        "prevalence": tune_threshold(oof_labels, oof_probs, "prevalence", prevalence=pool_prev),
    }
    log.info(f"OOF thresholds: {oof_thresholds}")
    best_t = oof_thresholds[args.threshold_metric]

    # --- 4: final ensemble on official train, dev for early stopping ---
    dev_loader = DataLoader(ChunkBagDataset(dev_ivs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ChunkBagDataset(test_ivs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)
    test_runs, dev_runs = [], []
    test_labels = dev_labels = None
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        model, _ = train_one_seed(args, seed, train_ivs, dev_ivs, device, embedding_dim)
        tp, test_labels = predict(model, test_loader, device)
        dp, dev_labels = predict(model, dev_loader, device)
        test_runs.append(tp); dev_runs.append(dp)
    test_avg = np.mean(test_runs, axis=0)
    dev_avg = np.mean(dev_runs, axis=0)

    # --- 5: evaluate test at every OOF threshold (headline = chosen metric) ---
    results = {
        "args": vars(args),
        "oof_auc": oof_auc,
        "oof_thresholds": oof_thresholds,
        "pool_prevalence": pool_prev,
        "test_by_oof_strategy": {},
        "oof_probs": oof_probs.tolist(),
        "oof_labels": oof_labels.tolist(),
        "test_probs": test_avg.tolist(),
        "test_labels": np.asarray(test_labels).tolist(),
        "test_prob_runs": [p.tolist() for p in test_runs],
    }
    # Literature-comparable (Burdisso 2023): per-seed test metrics, no ensemble.
    per_seed_test = [compute_metrics(test_labels, (p >= best_t).astype(int), p)
                     for p in test_runs]
    results["test_per_seed"] = per_seed_test
    for k in ("macro_f1", "f1", "roc_auc"):
        vals = [m[k] for m in per_seed_test]
        log.info(f"TEST per-seed {k} (t={best_t:.2f}): "
                 f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
    for name, t in oof_thresholds.items():
        tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        results["test_by_oof_strategy"][name] = {"threshold": t, **tm}
        log.info(f"TEST [OOF-{name}] (t={t:.2f}): "
                 f"F1={tm['f1']:.4f} P={tm['precision']:.4f} R={tm['recall']:.4f} "
                 f"BAcc={tm['balanced_accuracy']:.4f} AUC={tm['roc_auc']:.4f}")
    results["headline"] = results["test_by_oof_strategy"][args.threshold_metric]

    # Compare against tuning on the small dev for reference.
    dev_prev_t = tune_threshold(dev_labels, dev_avg, "prevalence",
                                prevalence=float(np.mean([iv["label"] for iv in train_ivs])))
    dm = compute_metrics(test_labels, (test_avg >= dev_prev_t).astype(int), test_avg)
    results["test_dev_prevalence_ref"] = {"threshold": dev_prev_t, **dm}
    log.info(f"TEST [dev-prevalence ref] (t={dev_prev_t:.2f}): "
             f"F1={dm['f1']:.4f} P={dm['precision']:.4f} R={dm['recall']:.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
