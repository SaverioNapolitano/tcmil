"""Multi-encoder TC-MIL ensemble on the official AVEC2017 protocol.

Trains TC-MIL independently on several frozen encoders and averages the
predicted probabilities. Encoders make different mistakes, so the averaged
ranking is sharper than any single one — which lifts F1 at any threshold,
not just AUC. Threshold is selected on dev only (default: prevalence-matched,
the strategy that is stable on the tiny 33-subject dev); test is evaluated
once and all strategies are logged for transparency.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.tcmil_data import assert_no_leakage, embed_chunks, load_official_split
from src.training.train_tcmil_official import (
    ChunkBagDataset,
    collate,
    predict,
    train_one_seed,
    tune_threshold,
)
from src.core.utils.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser(description="Multi-encoder TC-MIL ensemble (official protocol)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/tcmil_ensemble")
    p.add_argument("--encoders", nargs="+", default=[
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-mpnet-base-v2",
    ])
    p.add_argument("--temporal", default="none", choices=["none", "gru", "transformer"],
                   help="Context layer over the chunk sequence, applied to every member.")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
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
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=42)
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

    train_base = load_official_split(args.data_dir, "train", args.window, args.stride)
    dev_base = load_official_split(args.data_dir, "dev", args.window, args.stride)
    test_base = load_official_split(args.data_dir, "test", args.window, args.stride)
    assert_no_leakage(train_base, dev_base, test_base)
    train_prev = float(np.mean([iv["label"] for iv in train_base]))

    # Probabilities accumulated across every (encoder, seed) member.
    dev_members, test_members = [], []
    dev_labels = test_labels = None
    per_encoder = {}

    for enc in args.encoders:
        cache_tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
        # Fresh dicts so each encoder writes its own "embeddings" key.
        train_ivs = [dict(iv) for iv in train_base]
        dev_ivs = [dict(iv) for iv in dev_base]
        test_ivs = [dict(iv) for iv in test_base]
        for ivs in (train_ivs, dev_ivs, test_ivs):
            embed_chunks(ivs, enc, device, max_len=args.max_len, cache_tag=cache_tag)
        embedding_dim = train_ivs[0]["embeddings"].size(1)

        dev_loader = DataLoader(ChunkBagDataset(dev_ivs), batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate)
        test_loader = DataLoader(ChunkBagDataset(test_ivs), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate)

        enc_dev, enc_test, aucs = [], [], []
        for i in range(args.n_seeds):
            seed = args.base_seed + i
            model, _ = train_one_seed(args, seed, train_ivs, dev_ivs, device, embedding_dim)
            dp, dev_labels = predict(model, dev_loader, device)
            tp, test_labels = predict(model, test_loader, device)
            enc_dev.append(dp); enc_test.append(tp)
            dev_members.append(dp); test_members.append(tp)
            aucs.append(compute_metrics(dev_labels, (dp >= 0.5).astype(int), dp)["roc_auc"])

        enc_dev_avg = np.mean(enc_dev, axis=0)
        enc_test_avg = np.mean(enc_test, axis=0)
        d_auc = compute_metrics(dev_labels, (enc_dev_avg >= 0.5).astype(int), enc_dev_avg)["roc_auc"]
        t_auc = compute_metrics(test_labels, (enc_test_avg >= 0.5).astype(int), enc_test_avg)["roc_auc"]
        per_encoder[enc] = {"dev_auc": d_auc, "test_auc": t_auc,
                            "per_seed_dev_auc": [float(a) for a in aucs]}
        log.info(f"[{enc}] dev AUC={d_auc:.4f} test AUC={t_auc:.4f} "
                 f"(per-seed dev {np.mean(aucs):.4f}±{np.std(aucs):.4f})")

    dev_avg = np.mean(dev_members, axis=0)
    test_avg = np.mean(test_members, axis=0)

    strategies = {
        "f1": tune_threshold(dev_labels, dev_avg, "f1"),
        "bacc": tune_threshold(dev_labels, dev_avg, "bacc"),
        "youden": tune_threshold(dev_labels, dev_avg, "youden"),
        "prevalence": tune_threshold(dev_labels, dev_avg, "prevalence", prevalence=train_prev),
    }

    dev_by, test_by = {}, {}
    for name, t in strategies.items():
        dm = compute_metrics(dev_labels, (dev_avg >= t).astype(int), dev_avg)
        tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        dev_by[name] = {"threshold": t, **dm}
        test_by[name] = {"threshold": t, **tm}
        log.info(f"ENSEMBLE TEST [{name}] (t={t:.2f}): "
                 f"F1={tm['f1']:.4f} P={tm['precision']:.4f} R={tm['recall']:.4f} "
                 f"BAcc={tm['balanced_accuracy']:.4f} AUC={tm['roc_auc']:.4f}")

    headline = test_by[args.threshold_metric]
    log.info(f"HEADLINE [{args.threshold_metric}]: " +
             " ".join(f"{k}={v:.4f}" for k, v in headline.items() if isinstance(v, float)))

    results = {
        "args": vars(args),
        "per_encoder": per_encoder,
        "train_prevalence": train_prev,
        "dev_by_strategy": dev_by,
        "test_by_strategy": test_by,
        "headline": headline,
        "dev_probs": dev_avg.tolist(),
        "dev_labels": np.asarray(dev_labels).tolist(),
        "test_probs": test_avg.tolist(),
        "test_labels": np.asarray(test_labels).tolist(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
