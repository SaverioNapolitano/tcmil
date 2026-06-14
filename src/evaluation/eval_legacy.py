"""Legacy architectures under the current rigorous evaluation protocol.

Re-evaluates the pre-TC-MIL model family on the FULL dataset (189 raw
transcripts) with the same protocols as TC-MIL so the ablation table is
directly comparable:

    official - train 107 / dev-earlystop+threshold / test once, seed ensemble
    kfold    - StratifiedGroupKFold(5) over train+dev, inner-val earlystop,
               testprev threshold, per-fold seed ensemble
    mc       - StratifiedShuffleSplit(5) idem

Architecture ladder (the ablation axes vs TC-MIL):

    dialogue_mean   - no MIL: mean of every utterance embedding -> MLP
    flat_mil_mean   - utterance MIL, mean pooling (reimplemented: original
                      source survives only as a py3.11 .pyc)
    flat_mil_attn   - utterance MIL, gated-attention pooling (reimplemented)
    damil_r         - role-aware dual attention MIL (models/damil_r.py)
    ss_damil_r_v9   - + symptom supervision, multi-head pooling
                      (models/ss_damil_r.py, the v9d-official architecture)

Not ported (documented in the ablation study): damil_h (source not
recoverable), damil_x (requires the linguistic-feature pipeline),
damil_cl (contrastive training loop). All trick stacks (mixup/SWA/SAM/FGM)
are intentionally absent — prior ablations showed they add variance only.

Instances are utterance-level mpnet embeddings (the legacy representation);
TC-MIL differs by chunk-level instances + GRU. That contrast is the point.
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.dataset import load_interviews_with_roles
from src.core.models.damil_r import DAMILRClassifier
from src.core.models.ss_damil_r import (
    SSDamilRClassifierV9,
    SSDamilRClassifierV25,
    SSDamilRClassifierV26,
    SSDamilRClassifierV29,
)
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.evaluation import _seed_ensemble_metrics
from src.core.utils.metrics import compute_metrics
from src.core.utils.stats import compute_aggregate_metrics, format_aggregate_report

EMBED_CACHE = Path("cache/legacy_mpnet_utt")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Simple baselines (flat sources lost; faithful reimplementations)
# --------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, dim=768, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DialogueMean(nn.Module):
    """Non-MIL: mean of all utterance embeddings (both roles) -> MLP."""

    def __init__(self, dim=768):
        super().__init__()
        self.head = MLPHead(dim)

    def forward(self, patient, interviewer):
        return self.head(torch.cat([patient, interviewer], dim=0).mean(0))


class FlatMILMean(nn.Module):
    """Utterance MIL, mean pooling over patient instances."""

    def __init__(self, dim=768):
        super().__init__()
        self.head = MLPHead(dim)

    def forward(self, patient, interviewer):
        return self.head(patient.mean(0))


class FlatMILAttn(nn.Module):
    """Utterance MIL, gated-attention pooling (Ilse & Welling)."""

    def __init__(self, dim=768, proj_dim=128, attn_dim=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(proj_dim, attn_dim)
        self.U = nn.Linear(proj_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.cls = nn.Linear(proj_dim, 1)

    def forward(self, patient, interviewer):
        h = self.proj(patient)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1), 0)
        return self.cls((h * a.unsqueeze(-1)).sum(0)).squeeze(-1)


# --------------------------------------------------------------------------
# Adapters: name -> (ctor, forward -> (logit, aux_loss|None))
# --------------------------------------------------------------------------

def _fwd_simple(model, iv, train):
    logit = model(iv["patient_embeddings"], iv["interviewer_embeddings"])
    return logit, None


def _fwd_damil_r(model, iv, train):
    logit, _, _ = model(iv["patient_embeddings"], iv["interviewer_embeddings"],
                        noise_std=0.01 if train else 0.0)
    return logit, None


def _fwd_ss(model, iv, train):
    """Symptom-supervised SS-DAMIL-R family (V9/V25/V29): dict output with
    'logit' + 'symptom_logits', optional 'diversity_loss'."""
    out = model(iv["patient_embeddings"], iv["interviewer_embeddings"],
                noise_std=0.01 if train else 0.0)
    aux = None
    if iv["has_symptoms"]:
        target = (torch.tensor(iv["symptoms"],
                               device=out["symptom_logits"].device) >= 1).float()
        aux = 0.3 * nn.functional.binary_cross_entropy_with_logits(
            out["symptom_logits"], target)
        div = out.get("diversity_loss")
        if div is not None:
            aux = aux + 0.1 * div
    return out.get("logit", out.get("logits")), aux


def _fwd_ss_backbone(model, iv, train):
    """V29-style: forward_backbone -> forward_heads (no unified forward)."""
    pooled, div = model.forward_backbone(
        iv["patient_embeddings"], iv["interviewer_embeddings"],
        noise_std=0.01 if train else 0.0)
    out = model.forward_heads(pooled)
    logit = out.get("logit", out.get("logits"))
    aux = None
    if iv["has_symptoms"]:
        target = (torch.tensor(iv["symptoms"],
                               device=out["symptom_logits"].device) >= 1).float()
        aux = 0.3 * nn.functional.binary_cross_entropy_with_logits(
            out["symptom_logits"], target)
        if div is not None:
            aux = aux + 0.1 * div
    return logit, aux


REGISTRY = {
    "dialogue_mean": (lambda: DialogueMean(), _fwd_simple),
    "flat_mil_mean": (lambda: FlatMILMean(), _fwd_simple),
    "flat_mil_attn": (lambda: FlatMILAttn(), _fwd_simple),
    "damil_r": (lambda: DAMILRClassifier(embedding_dim=768, proj_dim=64,
                                         att_hidden_dim=32), _fwd_damil_r),
    "ss_damil_r_v9": (lambda: SSDamilRClassifierV9(embedding_dim=768,
                                                   proj_dim=64), _fwd_ss),
    # Distinct SS-DAMIL-R architectures (top non-V9 performers on the old
    # pooled-189 board): V25 gated symptom injection, V29 1D-conv front-end.
    "ss_damil_r_v25": (lambda: SSDamilRClassifierV25(embedding_dim=768,
                                                     proj_dim=64), _fwd_ss),
    # V26 ships a broken unified forward (unpacks 3, backbone yields 2);
    # drive it through the backbone like V29.
    "ss_damil_r_v26": (lambda: SSDamilRClassifierV26(embedding_dim=768,
                                                     proj_dim=64), _fwd_ss_backbone),
    "ss_damil_r_v29": (lambda: SSDamilRClassifierV29(embedding_dim=768,
                                                     proj_dim=64), _fwd_ss_backbone),
}


# --------------------------------------------------------------------------
# Embeddings (utterance-level mpnet, cached per interview)
# --------------------------------------------------------------------------

@torch.no_grad()
def embed_utterances(interviews, device, max_len=128, batch_size=32):
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    tok = enc = None
    for iv in interviews:
        cp = EMBED_CACHE / f"{iv['interview_id']}.pt"
        if cp.exists():
            d = torch.load(cp, weights_only=True)
            iv["patient_embeddings"] = d["patient"]
            iv["interviewer_embeddings"] = d["interviewer"]
            continue
        if enc is None:
            tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            enc = AutoModel.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2").to(device).eval()

        def embed(texts):
            if not texts:
                return torch.zeros(1, 768)
            outs = []
            for i in range(0, len(texts), batch_size):
                e = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
                h = enc(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).float()
                outs.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9),
                                        p=2, dim=1).cpu())
            return torch.cat(outs)

        iv["patient_embeddings"] = embed(iv["utterances"])
        iv["interviewer_embeddings"] = embed(iv["interviewer_utterances"])
        torch.save({"patient": iv["patient_embeddings"],
                    "interviewer": iv["interviewer_embeddings"]}, cp)
    return interviews


# --------------------------------------------------------------------------
# Train / predict one seed (bag-at-a-time, like the legacy scripts)
# --------------------------------------------------------------------------

def to_dev(iv, device):
    return {**iv,
            "patient_embeddings": iv["patient_embeddings"].to(device),
            "interviewer_embeddings": iv["interviewer_embeddings"].to(device)}


@torch.no_grad()
def predict(model, fwd, ivs, device):
    model.eval()
    probs = np.array([torch.sigmoid(fwd(model, to_dev(iv, device), False)[0]).item()
                      for iv in ivs])
    labels = np.array([iv["label"] for iv in ivs])
    return probs, labels


def train_one(name, seed, train_ivs, sel_ivs, device, args, log):
    set_seed(seed)
    ctor, fwd = REGISTRY[name]
    model = ctor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_pos = sum(iv["label"] for iv in train_ivs)
    pw = torch.tensor([(len(train_ivs) - n_pos) / max(1, n_pos)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = list(range(len(train_ivs)))
        random.shuffle(order)
        opt.zero_grad()
        for step, i in enumerate(order, 1):
            iv = to_dev(train_ivs[i], device)
            logit, aux = fwd(model, iv, True)
            loss = bce(logit.view(1), torch.tensor([float(iv["label"])], device=device))
            if aux is not None:
                loss = loss + aux
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        probs, labels = predict(model, fwd, sel_ivs, device)
        auc = compute_metrics(labels, (probs >= 0.5).astype(int), probs)["roc_auc"]
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, fwd


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------

def run_official(name, train_ivs, dev_ivs, test_ivs, device, args, log, out):
    train_prev = float(np.mean([iv["label"] for iv in train_ivs]))
    dev_runs, test_runs = [], []
    for s in range(args.n_seeds):
        model, fwd = train_one(name, args.seed + s, train_ivs, dev_ivs, device, args, log)
        dp, dev_labels = predict(model, fwd, dev_ivs, device)
        tp, test_labels = predict(model, fwd, test_ivs, device)
        dev_runs.append(dp)
        test_runs.append(tp)
        log.info(f"  [{name}] official seed {args.seed+s}: dev AUC="
                 f"{compute_metrics(dev_labels, (dp>=.5).astype(int), dp)['roc_auc']:.4f}")
    dev_avg, test_avg = np.mean(dev_runs, 0), np.mean(test_runs, 0)
    thresholds = {
        "f1": tune_threshold(dev_labels, dev_avg, "f1"),
        "prevalence": tune_threshold(dev_labels, dev_avg, "prevalence",
                                     prevalence=train_prev),
    }
    res = {"thresholds": thresholds, "test_by_strategy": {},
           "dev_probs": dev_avg.tolist(), "dev_labels": dev_labels.tolist(),
           "test_probs": test_avg.tolist(), "test_labels": test_labels.tolist(),
           "test_prob_runs": [p.tolist() for p in test_runs]}
    for k, t in thresholds.items():
        m = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        res["test_by_strategy"][k] = {"threshold": float(t), **m}
        log.info(f"  [{name}] TEST [{k}] t={t:.2f}: AUC={m['roc_auc']:.4f} "
                 f"F1={m['f1']:.4f} macroF1={m['macro_f1']:.4f}")
    t = thresholds["prevalence"]
    res["test_per_seed"] = [compute_metrics(test_labels, (p >= t).astype(int), p)
                            for p in test_runs]
    return res


def run_cv(name, pool, mode, device, args, log, out):
    labels = [iv["label"] for iv in pool]
    groups = [iv["interview_id"] for iv in pool]
    pool_prev = float(np.mean(labels))
    if mode == "kfold":
        split_iter = StratifiedGroupKFold(5, shuffle=True, random_state=args.seed)\
            .split(np.zeros(len(pool)), labels, groups)
    else:
        split_iter = StratifiedShuffleSplit(5, test_size=0.2, random_state=args.seed)\
            .split(np.zeros(len(pool)), labels)

    raw, group_preds = [], {}
    pool_np = np.array(pool)
    for fold, (tr_idx, te_idx) in enumerate(split_iter, 1):
        ftrain, ftest = pool_np[tr_idx].tolist(), pool_np[te_idx].tolist()
        assert {iv["interview_id"] for iv in ftest}.isdisjoint(
            {iv["interview_id"] for iv in ftrain}), "LEAKAGE"
        sids = sorted(iv["interview_id"] for iv in ftrain)
        lab = {iv["interview_id"]: iv["label"] for iv in ftrain}
        tr_i, va_i = next(StratifiedShuffleSplit(1, test_size=0.15,
                          random_state=args.seed + fold)
                          .split(np.zeros(len(sids)), [lab[s] for s in sids]))
        va = {sids[i] for i in va_i}
        itrain = [iv for iv in ftrain if iv["interview_id"] not in va]
        ival = [iv for iv in ftrain if iv["interview_id"] in va]

        group_preds[fold] = []
        for s in range(args.cv_seeds):
            model, fwd = train_one(name, args.seed + fold * 100 + s,
                                   itrain, ival, device, args, log)
            tp, tl = predict(model, fwd, ftest, device)
            vp, vl = predict(model, fwd, ival, device)
            t = tune_threshold(None, tp, metric="prevalence", prevalence=pool_prev)
            m = compute_metrics(tl, (tp >= t).astype(int), tp)
            m.update({"_fold_idx": fold, "_seed_idx": s,
                      "probability": tp.tolist(), "true_label": tl.tolist(),
                      "val_probability": vp.tolist(), "val_true_label": vl.tolist()})
            raw.append(m)
            group_preds[fold].append({"probability": tp, "true_label": tl,
                                      "val_probability": vp, "val_true_label": vl})
            log.info(f"  [{name}] {mode} f{fold}s{s}: AUC={m['roc_auc']:.4f} "
                     f"F1={m['f1']:.4f}")

    def ens_thr(gi, vy, vp, tp):
        return tune_threshold(None, tp, metric="prevalence", prevalence=pool_prev)

    ens = _seed_ensemble_metrics(group_preds, "_fold_idx", ens_thr)
    return {"per_run_aggregate": compute_aggregate_metrics(raw),
            "ensemble_aggregate": compute_aggregate_metrics(ens),
            "raw": raw}


def main():
    p = argparse.ArgumentParser(description="Legacy architectures, rigorous protocol")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_root", default="results/legacy_aligned")
    p.add_argument("--models", nargs="+", default=list(REGISTRY.keys()))
    p.add_argument("--protocols", nargs="+", default=["official", "kfold", "mc"])
    p.add_argument("--n_seeds", type=int, default=5)       # official
    p.add_argument("--cv_seeds", type=int, default=3)      # per fold/split
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(root / "run.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    splits = {s: load_interviews_with_roles(args.data_dir, s, use_raw_data=True)
              for s in ("train", "dev", "test")}
    ids = [{iv["interview_id"] for iv in splits[s]} for s in ("train", "dev", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) \
        and ids[1].isdisjoint(ids[2]), "LEAKAGE: official splits overlap"
    for s, ivs in splits.items():
        embed_utterances(ivs, device)
        log.info(f"{s}: {len(ivs)} interviews embedded")
    pool = splits["train"] + splits["dev"]

    for name in args.models:
        for proto in args.protocols:
            out = root / f"{name}_{proto}"
            out.mkdir(parents=True, exist_ok=True)
            log.info(f"=== {name} / {proto} ===")
            if proto == "official":
                res = run_official(name, splits["train"], splits["dev"],
                                   splits["test"], device, args, log, out)
            else:
                res = run_cv(name, pool, proto, device, args, log, out)
                log.info("\n## Ensemble\n" +
                         format_aggregate_report(res["ensemble_aggregate"]))
            with open(out / "results.json", "w") as f:
                json.dump({"args": vars(args), "model": name, "protocol": proto, **res},
                          f, indent=2,
                          default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
            log.info(f"saved {out}")


if __name__ == "__main__":
    main()
