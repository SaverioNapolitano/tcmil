"""Multi-granularity bags (ranking lever, single model, no FT).

Builds each interview's bag as the UNION of chunks at several window sizes
(default w2+w4+w6), so the MIL head sees the same dialogue at multiple
granularities -> potentially better instance separation -> higher AUC ->
higher oracle ceiling. Same frozen encoder, pos_weight=1.0. Reuses the official
training loop; reports test vs the single bar 0.774 / AUC 0.864.

Honest note: the per-window design ablation barely moved the score, so expect a
small effect; this checks whether the *union* helps beyond any single window.
Saves per-seed checkpoints + probs (save-everything).
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
from src.core.tcmil_data import load_official_split, embed_chunks, assert_no_leakage
from src.core.utils.metrics import compute_metrics
from src.training.train_tcmil_official import (ChunkBagDataset, collate, predict,
                                           train_one_seed, tune_threshold)

PREV, BAR = 0.28, 0.774


def merged_split(data_dir, split, windows, strides, max_len, encoder, device):
    """Load each granularity, concat per-interview chunk embeddings into one bag."""
    base = None
    for w, s in zip(windows, strides):
        ivs = load_official_split(data_dir, split, w, s)
        embed_chunks(ivs, encoder, device, max_len=max_len, cache_tag=f"_w{w}_s{s}_l{max_len}")
        if base is None:
            base = {iv["interview_id"]: dict(iv) for iv in ivs}
            for iv in ivs:
                base[iv["interview_id"]]["embeddings"] = iv["embeddings"]
        else:
            for iv in ivs:
                b = base[iv["interview_id"]]
                b["embeddings"] = torch.cat([b["embeddings"], iv["embeddings"]], dim=0)
                b["chunks"] = b["chunks"] + iv["chunks"]
    return list(base.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--windows", default="2,4,6")
    ap.add_argument("--strides", default="1,2,3")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--temporal", default="gru")
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--attn_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--instance_dropout", type=float, default=0.1)
    ap.add_argument("--noise_std", type=float, default=0.02)
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--aux_mode", default="binary")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--base_seed", type=int, default=100)
    ap.add_argument("--output_dir", default="results/single_model/multigran_w246")
    args = ap.parse_args()
    args.windows = [int(x) for x in args.windows.split(",")]
    args.strides = [int(x) for x in args.strides.split(",")]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__); log.info(f"Args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    train = merged_split("data", "train", args.windows, args.strides, args.max_len, args.encoder_name, device)
    dev = merged_split("data", "dev", args.windows, args.strides, args.max_len, args.encoder_name, device)
    test = merged_split("data", "test", args.windows, args.strides, args.max_len, args.encoder_name, device)
    assert_no_leakage(train, dev, test)
    dim = train[0]["embeddings"].size(1)
    log.info(f"train={len(train)} dev={len(dev)} test={len(test)} dim={dim} "
             f"mean_chunks={np.mean([len(iv['chunks']) for iv in train]):.1f}")

    ckpt = out / "checkpoints"; ckpt.mkdir(exist_ok=True)
    json.dump({"embedding_dim": dim, **{k: getattr(args, k) for k in
              ("proj_dim","attn_dim","temporal","gru_layers","dropout","encoder_name","windows","strides")}},
              open(ckpt / "config.json", "w"), indent=2)
    dev_ld = DataLoader(ChunkBagDataset(dev), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_ld = DataLoader(ChunkBagDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    dev_runs, test_runs, ytest = [], [], None
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        model, _ = train_one_seed(args, seed, train, dev, device, dim)
        torch.save(model.state_dict(), ckpt / f"seed_{seed}.pt")
        dp, _ = predict(model, dev_ld, device); dev_runs.append(dp)
        tp, ytest = predict(model, test_ld, device); test_runs.append(tp)
        log.info(f"seed {seed} done")

    em = []
    for r in test_runs:
        t = tune_threshold(None, r, metric="prevalence", prevalence=PREV)
        em.append(compute_metrics(ytest, (r >= t).astype(int), r))
    mac = np.array([m["macro_f1"] for m in em])
    mic = np.array([m["micro_f1"] for m in em]); auc = np.array([m["roc_auc"] for m in em])
    log.info(f"TEST full-{args.n_seeds}: macro {mac.mean():.3f}±{mac.std():.3f} "
             f"micro {mic.mean():.3f} auc {auc.mean():.3f}  (single bar {BAR}, AUC 0.864)")
    json.dump({"args": vars(args), "test_labels": ytest.tolist(),
               "test_prob_runs": [r.tolist() for r in test_runs],
               "dev_prob_runs": [r.tolist() for r in dev_runs]},
              open(out / "results.json", "w"), indent=2)
    log.info(f"saved to {out}")


if __name__ == "__main__":
    main()
