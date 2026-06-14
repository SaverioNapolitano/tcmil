"""B6 — Negative-Correlation Learning (NCL) for the 2-encoder ensemble.

Jointly trains two TC-MIL members (different encoders, e.g. bge-large + UAE-large)
with a coupling penalty that REWARDS disagreement, so their errors decorrelate
and the averaged ensemble improves. Standard NCL (Liu & Yao): each member m adds
lambda * p_m_penalty, with p_m_penalty = (p_m - p_ens) * sum_{k!=m}(p_k - p_ens).
For two members this reduces to total term  2*lambda*(p_a-p_ens)*(p_b-p_ens)
= -(lambda/2)*(p_a-p_b)^2  -> minimizing the loss pushes the members apart.

Both members are frozen-encoder TC-MIL heads over their own encoder's cached
chunk embeddings (same interviews/order so the coupling is per-bag aligned).
Selection (early stop) + threshold on dev only; test once with --eval_test.

Saves EVERYTHING: per-seed checkpoints for BOTH heads, per-member and ensemble
dev/test prob runs, labels, config. Reports ensemble at the prevalence threshold
(full 30-seed + 6x(5+5)) vs the independent-pair baseline (0.833) and the single
bar 0.774.
"""

import argparse
import json
import logging
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
from src.training.train_tcmil_official import set_seed, tune_threshold

PREV, MIL_BAR, SINGLE_BAR = 0.28, 0.739, 0.774


class PairBags(Dataset):
    """Returns both encoders' bags for the same interview (aligned)."""
    def __init__(self, ivs):
        self.ivs = ivs

    def __len__(self):
        return len(self.ivs)

    def __getitem__(self, i):
        iv = self.ivs[i]
        sym = torch.tensor(iv["symptoms"], dtype=torch.float)
        return {"a": iv["emb_a"], "b": iv["emb_b"],
                "label": torch.tensor(iv["label"], dtype=torch.float),
                "symptoms_bin": (sym >= 1).float(),
                "has_symptoms": torch.tensor(float(iv["has_symptoms"]))}


def _pad(bags):
    n = max(b.size(0) for b in bags); d = bags[0].size(1)
    p = torch.zeros(len(bags), n, d); m = torch.zeros(len(bags), n)
    for i, b in enumerate(bags):
        p[i, :b.size(0)] = b; m[i, :b.size(0)] = 1
    return p, m


def collate(batch):
    pa, ma = _pad([x["a"] for x in batch])
    pb, mb = _pad([x["b"] for x in batch])
    return {"a": pa, "ma": ma, "b": pb, "mb": mb,
            "labels": torch.stack([x["label"] for x in batch]),
            "symptoms_bin": torch.stack([x["symptoms_bin"] for x in batch]),
            "has_symptoms": torch.stack([x["has_symptoms"] for x in batch])}


@torch.no_grad()
def predict_pair(ma, mb, loader, device):
    ma.eval(); mb.eval()
    pa, pb, y = [], [], []
    for batch in loader:
        oa = ma(batch["a"].to(device), batch["ma"].to(device))
        ob = mb(batch["b"].to(device), batch["mb"].to(device))
        pa.extend(torch.sigmoid(oa["logits"]).cpu().numpy().tolist())
        pb.extend(torch.sigmoid(ob["logits"]).cpu().numpy().tolist())
        y.extend(batch["labels"].numpy().tolist())
    return np.array(pa), np.array(pb), np.array(y)


def build(dim, args, temporal, device):
    return TCMIL(embedding_dim=dim, proj_dim=args.proj_dim, attn_dim=args.attn_dim,
                 dropout=args.dropout, temporal=temporal,
                 gru_layers=args.gru_layers).to(device)


def aux_term(out, batch, sym_loss, device, w):
    if w <= 0:
        return 0.0
    per = sym_loss(out["symptom_logits"], batch["symptoms_bin"].to(device))
    has = batch["has_symptoms"].to(device).unsqueeze(1)
    denom = has.sum() * 8
    return w * (per * has).sum() / denom if denom > 0 else 0.0


def train_seed(args, seed, train_ivs, dev_ivs, dims, device):
    set_seed(seed)
    tl = DataLoader(PairBags(train_ivs), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl = DataLoader(PairBags(dev_ivs), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    ma = build(dims[0], args, args.temporal_a, device)
    mb = build(dims[1], args, args.temporal_b, device)
    pw = torch.tensor([args.pos_weight]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    sym_loss = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(list(ma.parameters()) + list(mb.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    best, best_states, bad = -1.0, None, 0
    for _ in range(args.max_epochs):
        ma.train(); mb.train()
        for batch in tl:
            opt.zero_grad()
            y = batch["labels"].to(device)
            oa = ma(batch["a"].to(device), batch["ma"].to(device))
            ob = mb(batch["b"].to(device), batch["mb"].to(device))
            la = bce(oa["logits"], y) + aux_term(oa, batch, sym_loss, device, args.aux_weight)
            lb = bce(ob["logits"], y) + aux_term(ob, batch, sym_loss, device, args.aux_weight)
            pa, pb = torch.sigmoid(oa["logits"]), torch.sigmoid(ob["logits"])
            ens = (pa + pb) / 2
            ncl = ((pa - ens) * (pb - ens)).mean()          # = -(pa-pb)^2/4
            loss = la + lb + 2 * args.ncl_lambda * ncl       # minimize -> spread members
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ma.parameters()) + list(mb.parameters()), 1.0)
            opt.step()
        pa, pb, yv = predict_pair(ma, mb, dl, device)
        auc = compute_metrics(yv, ((pa + pb) / 2 >= 0.5).astype(int), (pa + pb) / 2)["roc_auc"]
        if auc > best:
            best, bad = auc, 0
            best_states = ({k: v.detach().cpu().clone() for k, v in ma.state_dict().items()},
                           {k: v.detach().cpu().clone() for k, v in mb.state_dict().items()})
        else:
            bad += 1
        if bad >= args.patience:
            break
    ma.load_state_dict(best_states[0]); mb.load_state_dict(best_states[1])
    return ma, mb


def main():
    p = argparse.ArgumentParser(description="B6 NCL 2-encoder ensemble")
    p.add_argument("--encoder_a", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--encoder_b", default="WhereIsAI/UAE-Large-V1")
    p.add_argument("--temporal_a", default="gru")
    p.add_argument("--temporal_b", default="gru")
    p.add_argument("--ncl_lambda", type=float, default=0.5)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/ensemble/ncl_bge_uae")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--aux_weight", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--base_seed", type=int, default=100)
    p.add_argument("--eval_test", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__); log.info(f"Args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    splits = {s: load_official_split(args.data_dir, s, args.window, args.stride)
              for s in ("train", "dev", "test")}
    assert_no_leakage(splits["train"], splits["dev"], splits["test"])
    tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
    for key, enc in (("emb_a", args.encoder_a), ("emb_b", args.encoder_b)):
        for ivs in splits.values():
            embed_chunks(ivs, enc, device, max_len=args.max_len, cache_tag=tag)
            for iv in ivs:
                iv[key] = iv["embeddings"]
    dims = (splits["train"][0]["emb_a"].size(1), splits["train"][0]["emb_b"].size(1))
    log.info(f"dims a={dims[0]} b={dims[1]}")

    ckpt = out / "checkpoints"; ckpt.mkdir(exist_ok=True)
    json.dump({**vars(args), "dim_a": dims[0], "dim_b": dims[1]},
              open(ckpt / "config.json", "w"), indent=2)
    dev_ld = DataLoader(PairBags(splits["dev"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_ld = DataLoader(PairBags(splits["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    a_dev, b_dev, ens_dev = [], [], []
    a_test, b_test, ens_test = [], [], []
    ydev = ytest = None
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        ma, mb = train_seed(args, seed, splits["train"], splits["dev"], dims, device)
        torch.save(ma.state_dict(), ckpt / f"seed_{seed}_A.pt")
        torch.save(mb.state_dict(), ckpt / f"seed_{seed}_B.pt")
        pa, pb, ydev = predict_pair(ma, mb, dev_ld, device)
        a_dev.append(pa); b_dev.append(pb); ens_dev.append((pa + pb) / 2)
        m = compute_metrics(ydev, ((pa + pb) / 2 >= 0.5).astype(int), (pa + pb) / 2)
        log.info(f"seed {seed}: dev ens AUC={m['roc_auc']:.4f}")
        if args.eval_test:
            pa, pb, ytest = predict_pair(ma, mb, test_ld, device)
            a_test.append(pa); b_test.append(pb); ens_test.append((pa + pb) / 2)

    def report(runs, y, tag):
        n = len(runs); em = []
        for r in runs:
            t = tune_threshold(None, r, metric="prevalence", prevalence=PREV)
            em.append(compute_metrics(y, (r >= t).astype(int), r))
        g = n // 5; grp = []
        for gi in range(g):
            prob = np.mean(runs[gi*5:gi*5+5], axis=0)
            t = tune_threshold(None, prob, metric="prevalence", prevalence=PREV)
            grp.append(compute_metrics(y, (prob >= t).astype(int), prob))
        blocks = [("full", em)] + ([("6x(5+5)", grp)] if g else [])
        for lab, ms in blocks:
            v = {k: np.array([m[k] for m in ms]) for k in ("macro_f1", "micro_f1", "roc_auc")}
            log.info(f"  {tag} {lab}: macro {v['macro_f1'].mean():.3f}±{v['macro_f1'].std():.3f} "
                     f"micro {v['micro_f1'].mean():.3f} auc {v['roc_auc'].mean():.3f}")
        return em

    res = {"args": vars(args), "dev_labels": ydev.tolist(),
           "memberA_dev_prob_runs": [r.tolist() for r in a_dev],
           "memberB_dev_prob_runs": [r.tolist() for r in b_dev]}
    if args.eval_test:
        corr = float(np.mean([np.corrcoef(a_test[i], b_test[i])[0, 1] for i in range(len(a_test))]))
        log.info(f"lambda={args.ncl_lambda} test member prob-corr={corr:.3f} "
                 f"(independent pair was 0.960)")
        report(ens_test, ytest, "ENSEMBLE")
        res.update({"test_labels": ytest.tolist(),
                    "memberA_test_prob_runs": [r.tolist() for r in a_test],
                    "memberB_test_prob_runs": [r.tolist() for r in b_test],
                    "test_member_corr": corr})
    json.dump(res, open(out / "results.json", "w"), indent=2)
    log.info(f"saved to {out}")


if __name__ == "__main__":
    main()
