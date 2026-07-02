"""MC-dropout averaging on a checkpointed model dir (probability-quality lever).

For each saved per-seed checkpoint, run T stochastic forward passes with dropout
kept ON, average -> MC probabilities. Compares MC vs the deterministic probs on
macro-F1 (prevalence threshold) and AUC, on dev and test. Honest test of whether
test-time stochastic averaging sharpens the ranking/calibration.

Needs <dir>/checkpoints/{config.json, seed_*.pt} (saved by train_tcmil_official
after 2026-06-14).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.models.tcmil import TCMIL
from src.core.tcmil_data import load_official_split, embed_chunks
from src.core.utils.metrics import compute_metrics
from src.training.train_tcmil_official import (ChunkBagDataset, collate, tune_threshold, set_seed)
from torch.utils.data import DataLoader

PREV = 0.28


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, loader, device, T):
    model.eval(); enable_mc_dropout(model)
    runs = []
    for _ in range(T):
        probs = []
        for batch in loader:
            out = model(batch["bags"].to(device), batch["mask"].to(device))
            probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().tolist())
        runs.append(np.array(probs))
    return np.mean(runs, axis=0)


@torch.no_grad()
def det_predict(model, loader, device):
    model.eval()
    probs, y = [], []
    for batch in loader:
        out = model(batch["bags"].to(device), batch["mask"].to(device))
        probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().tolist())
        y.extend(batch["labels"].numpy().tolist())
    return np.array(probs), np.array(y)


def macro_auc(runs, y):
    em = []
    for r in runs:
        t = tune_threshold(None, r, metric="prevalence", prevalence=PREV)
        em.append(compute_metrics(y, (r >= t).astype(int), r))
    def ms(k): v = [m[k] for m in em]; return np.mean(v), np.std(v)
    return ms("macro_f1"), ms("micro_f1"), ms("roc_auc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/single_model/headline_pw1_30seed")
    ap.add_argument("--T", type=int, default=30)
    args = ap.parse_args()
    cfg = json.load(open(Path(args.dir) / "checkpoints" / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ivs = {s: load_official_split("data", s, cfg["window"], cfg["stride"]) for s in ("dev", "test")}
    tag = f"_w{cfg['window']}_s{cfg['stride']}_l256"
    for s in ivs:
        embed_chunks(ivs[s], cfg["encoder_name"], device, max_len=256, cache_tag=tag)
    loaders = {s: DataLoader(ChunkBagDataset(ivs[s]), batch_size=16, shuffle=False, collate_fn=collate)
               for s in ivs}

    ckpts = sorted(Path(args.dir, "checkpoints").glob("seed_*.pt"))
    det_test, mc_test = [], []
    ytest = None
    for c in ckpts:
        set_seed(int(c.stem.split("_")[1]))
        model = TCMIL(embedding_dim=cfg["embedding_dim"], proj_dim=cfg["proj_dim"],
                      attn_dim=cfg["attn_dim"], dropout=cfg["dropout"],
                      temporal=cfg["temporal"], gru_layers=cfg["gru_layers"]).to(device)
        model.load_state_dict(torch.load(c, map_location=device))
        d, ytest = det_predict(model, loaders["test"], device)
        det_test.append(d)
        mc_test.append(mc_predict(model, loaders["test"], device, args.T))

    d_ma, d_mi, d_au = macro_auc(det_test, ytest)
    m_ma, m_mi, m_au = macro_auc(mc_test, ytest)
    print(f"checkpoints: {len(ckpts)}  T={args.T}")
    print(f"  deterministic: macro {d_ma[0]:.3f}±{d_ma[1]:.3f}  micro {d_mi[0]:.3f}±{d_mi[1]:.3f}  auc {d_au[0]:.3f}±{d_au[1]:.3f}")
    print(f"  MC-dropout   : macro {m_ma[0]:.3f}±{m_ma[1]:.3f}  micro {m_mi[0]:.3f}±{m_mi[1]:.3f}  auc {m_au[0]:.3f}±{m_au[1]:.3f}")
    print(f"  Δ macro {m_ma[0]-d_ma[0]:+.3f}  Δ auc {m_au[0]-d_au[0]:+.3f}  "
          f"({'gain' if m_ma[0] > d_ma[0] else 'no gain'})")


if __name__ == "__main__":
    main()
