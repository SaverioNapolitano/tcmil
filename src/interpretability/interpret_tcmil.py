"""Interpretability analysis for TC-MIL — as rigorous as the eval protocol.

TC-MIL exposes three interpretable surfaces: (1) the gated-attention weights
over dialogue chunks (which parts of the interview drive the decision),
(2) the PHQ-8 symptom auxiliary head (a clinical read-out), and (3) the
chunk instances themselves (human-readable dialogue). This script turns all
three into faithful, quantified, leakage-aware analyses rather than a single
cherry-picked attention heat-map.

Methodology (grounded in the literature, adapted to this project):

1. ATTENTION, honestly. Raw gated-attention weights per chunk (Ilse &
   Welling 2018). Reported with cross-seed *stability* (mean pairwise
   Spearman over the seed ensemble) — attention that changes every seed is
   not an explanation.

2. FAITHFULNESS, not just saliency (Jain & Wallace 2019; DeYoung et al.
   ERASER 2020). Attention is validated against occlusion:
     - comprehensiveness@k: Δprob when the top-k attended chunks are removed
       (high ⇒ they mattered),
     - sufficiency@k: Δprob when only the top-k are kept (low ⇒ they suffice),
     - attention/occlusion rank correlation: Spearman between each chunk's
       attention and its leave-one-out Δprob (positive ⇒ attention is faithful).

3. CLINICAL read-out. The PHQ-8 symptom head is scored per item (AUC across
   the split) and per interview (predicted vs ground-truth profile) — does
   the text actually carry each symptom?

4. ATTENTION ↔ SYMPTOM link (unique to this project). For the most-attended
   chunk, which PHQ-8 item does the symptom head most activate? Ties the main
   decision to a specific clinical construct.

5. VALIDITY / BIAS probes (Burdisso et al. 2024). The DAIC-WOZ text shortcut
   lives in interviewer prompts and the second interview half. We measure:
     - position bias: attention vs normalized chunk position,
     - interviewer-content bias: attention mass on chunks whose interviewer
       line contains PHQ-8-symptom keywords,
   and (with --compare_participant_only) contrast a dialogue model against a
   participant-only model.

Leakage: models are trained on the official train split only; analysis runs
on dev by default (--split test for the final, report-once figures). No
analysis statistic is fed back into training or thresholds.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.models.tcmil import TCMIL
from src.core.tcmil_data import assert_no_leakage, embed_chunks, load_official_split
from src.training.train_tcmil_official import set_seed, train_one_seed
from src.core.utils.metrics import compute_metrics

# PHQ-8-aligned symptom lexicon for the interviewer-content bias probe.
SYMPTOM_KEYWORDS = [
    "sleep", "sleeping", "insomnia", "tired", "fatigue", "energy",
    "eat", "eating", "appetite", "weight",
    "depress", "sad", "down", "mood", "hopeless",
    "interest", "enjoy", "pleasure", "motivat",
    "concentrat", "focus", "decision", "memory",
    "fail", "failure", "guilt", "worthless", "letdown",
    "slow", "restless", "fidget", "agitat",
    "hurt", "harm", "suicide", "dead", "kill",
    "diagnos", "depression", "ptsd", "therap", "medication",
]
PHQ8_ITEMS = ["NoInterest", "Depressed", "Sleep", "Tired",
              "Appetite", "Failure", "Concentrating", "Moving"]


@torch.no_grad()
def forward_full(model, bag, device):
    """Return (prob, attention[N]) for a single bag of chunk embeddings."""
    mask = torch.ones(1, bag.size(0), device=device)
    out = model(bag.unsqueeze(0).to(device), mask)
    return torch.sigmoid(out["logits"]).item(), out["attention"][0].cpu().numpy()


@torch.no_grad()
def forward_masked(model, bag, keep_idx, device):
    """Prob with only `keep_idx` chunks active (others masked out)."""
    mask = torch.zeros(1, bag.size(0), device=device)
    mask[0, keep_idx] = 1.0
    out = model(bag.unsqueeze(0).to(device), mask)
    return torch.sigmoid(out["logits"]).item()


@torch.no_grad()
def per_chunk_symptoms(model, bag, device):
    """Symptom-head logits applied to each chunk's contextualized rep.
    Returns (N, 8) — what PHQ-8 signal each chunk alone carries."""
    h = model.projector(bag.unsqueeze(0).to(device))
    if model.temporal == "gru":
        lengths = torch.tensor([bag.size(0)])
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False)
        ctx, _ = model.context(packed)
        ctx, _ = torch.nn.utils.rnn.pad_packed_sequence(
            ctx, batch_first=True, total_length=h.size(1))
        h = model.context_norm(h + ctx)
    elif model.temporal == "transformer":
        h = model.context(h)
    normed = model.norm(h[0])                  # (N, d)
    return model.symptom_head(normed).cpu().numpy()


def chunk_has_symptom_kw(chunk_text: str) -> bool:
    """Does the chunk's Interviewer line(s) mention a PHQ-8 keyword?"""
    iv_lines = [ln for ln in chunk_text.splitlines() if ln.startswith("Interviewer:")]
    blob = " ".join(iv_lines).lower()
    return any(kw in blob for kw in SYMPTOM_KEYWORDS)


def faithfulness(model, bag, attn, device, ks=(1, 2, 3, 5)):
    """ERASER comprehensiveness/sufficiency + attention/occlusion correlation."""
    n = bag.size(0)
    full = forward_full(model, bag, device)[0]
    order = np.argsort(-attn)                       # most-attended first
    comp, suff = {}, {}
    for k in ks:
        k = min(k, n)
        topk = order[:k]
        rest = order[k:]
        comp[k] = full - (forward_masked(model, bag, rest, device) if len(rest) else 0.0)
        suff[k] = full - forward_masked(model, bag, topk, device)
    # Leave-one-out: drop each chunk, measure Δprob; correlate with attention.
    if n >= 4:
        loo = np.array([full - forward_masked(model, bag,
                        [j for j in range(n) if j != i], device) for i in range(n)])
        rho = spearmanr(attn, loo).correlation
    else:
        loo = None
        rho = np.nan
    # loo[j] is the leave-one-out drop in p for chunk j; returned so the caller
    # can save the per-chunk (attention, occlusion-delta) pairs for the scatter.
    return comp, suff, (float(rho) if rho == rho else np.nan), loo


def analyze(args):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.INFO)
    log = logging.getLogger(__name__)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    train = load_official_split(args.data_dir, "train", args.window, args.stride,
                                participant_only=args.participant_only)
    dev = load_official_split(args.data_dir, "dev", args.window, args.stride,
                              participant_only=args.participant_only)
    test = load_official_split(args.data_dir, "test", args.window, args.stride,
                               participant_only=args.participant_only)
    assert_no_leakage(train, dev, test)
    target = {"dev": dev, "test": test}[args.split]

    tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
    if args.participant_only:
        tag += "_ponly"
    # dev is always needed (early-stopping signal in train_one_seed), even when
    # the analysis target is test.
    for ivs in {id(x): x for x in (train, dev, target)}.values():
        embed_chunks(ivs, args.encoder_name, device, max_len=args.max_len, cache_tag=tag)
    emb_dim = train[0]["embeddings"].size(1)

    # Train the seed ensemble on train; dev is only early-stopping signal.
    models = []
    for i in range(args.n_seeds):
        set_seed(args.base_seed + i)
        m, _ = train_one_seed(args, args.base_seed + i, train, dev, device, emb_dim)
        m.eval()
        models.append(m)
        log.info(f"trained interpretability seed {args.base_seed + i}")

    per_iv, stab_all, rho_all = [], [], []
    pos_attn_corr, kw_mass_all = [], []
    sym_pred_all, sym_true_all = [], []
    comp_agg = {k: [] for k in (1, 2, 3, 5)}
    suff_agg = {k: [] for k in (1, 2, 3, 5)}
    attn_sym_link = {it: 0 for it in PHQ8_ITEMS}

    for iv in target:
        bag = iv["embeddings"]
        n = bag.size(0)
        seed_attn = np.stack([forward_full(m, bag, device)[1] for m in models])  # (S, N)
        seed_prob = np.array([forward_full(m, bag, device)[0] for m in models])
        attn = seed_attn.mean(0)
        prob = float(seed_prob.mean())

        # Cross-seed attention stability (mean pairwise Spearman).
        if args.n_seeds > 1 and n >= 4:
            rhos = [spearmanr(seed_attn[a], seed_attn[b]).correlation
                    for a in range(args.n_seeds) for b in range(a + 1, args.n_seeds)]
            stab = float(np.nanmean(rhos))
            stab_all.append(stab)
        else:
            stab = np.nan

        # Faithfulness on seed 0 (representative; cheap and deterministic).
        comp, suff, rho, loo = faithfulness(models[0], bag, seed_attn[0], device)
        for k in comp:
            comp_agg[k].append(comp[k]); suff_agg[k].append(suff[k])
        if rho == rho:
            rho_all.append(rho)

        # Position bias: attention vs normalized chunk position.
        pos = np.arange(n) / max(1, n - 1)
        if n >= 4:
            pc = spearmanr(attn, pos).correlation
            if pc == pc:
                pos_attn_corr.append(pc)

        # Interviewer-content bias: attention mass on symptom-keyword chunks.
        kw = np.array([chunk_has_symptom_kw(c) for c in iv["chunks"][:n]], dtype=float)
        kw_mass_all.append(float((attn * kw).sum()))

        # Attention ↔ symptom link on the top attended chunk (seed 0).
        psym = per_chunk_symptoms(models[0], bag, device)        # (N, 8)
        top = int(np.argmax(seed_attn[0]))
        attn_sym_link[PHQ8_ITEMS[int(np.argmax(psym[top]))]] += 1

        # Clinical read-out: bag-level predicted symptom profile.
        with torch.no_grad():
            mask = torch.ones(1, n, device=device)
            sym_logits = models[0](bag.unsqueeze(0).to(device), mask)["symptom_logits"]
        sym_pred = torch.sigmoid(sym_logits)[0].cpu().numpy()
        sym_pred_all.append(sym_pred)
        sym_true_all.append([1 if s >= 1 else 0 for s in iv["symptoms"]])

        top_chunks = np.argsort(-attn)[:3].tolist()
        per_iv.append({
            "interview_id": iv["interview_id"], "label": iv["label"],
            "prob": prob, "n_chunks": n, "attn_stability": stab,
            "attn_occlusion_rho": rho,
            "top_chunks": [{"idx": j, "attn": float(attn[j]),
                            "text": iv["chunks"][j]} for j in top_chunks],
            "per_chunk": ([{"attn": float(attn[j]), "occ_delta": float(loo[j]),
                            "text": iv["chunks"][j]}
                           for j in range(n)] if loo is not None else []),
            "pred_symptoms": sym_pred.tolist(),
            "true_symptoms": [1 if s >= 1 else 0 for s in iv["symptoms"]],
        })

    # Per-symptom AUC across the split.
    sp = np.array(sym_pred_all); st = np.array(sym_true_all)
    sym_auc = {}
    for i, it in enumerate(PHQ8_ITEMS):
        if len(np.unique(st[:, i])) > 1:
            from sklearn.metrics import roc_auc_score
            sym_auc[it] = float(roc_auc_score(st[:, i], sp[:, i]))

    summary = {
        "split": args.split, "n_interviews": len(target), "n_seeds": args.n_seeds,
        "participant_only": args.participant_only,
        "attention_stability_mean": float(np.nanmean(stab_all)) if stab_all else None,
        "faithfulness": {
            "attn_occlusion_rho_mean": float(np.nanmean(rho_all)) if rho_all else None,
            "comprehensiveness": {k: float(np.mean(v)) for k, v in comp_agg.items()},
            "sufficiency": {k: float(np.mean(v)) for k, v in suff_agg.items()},
        },
        "bias_probes": {
            "position_attention_rho_mean": float(np.nanmean(pos_attn_corr)) if pos_attn_corr else None,
            "interviewer_keyword_attention_mass_mean": float(np.mean(kw_mass_all)),
        },
        "symptom_head_auc": sym_auc,
        "attention_symptom_link": attn_sym_link,
    }

    with open(out_dir / f"interpret_{args.split}.json", "w") as f:
        json.dump({"summary": summary, "per_interview": per_iv}, f, indent=2)
    log.info("SUMMARY\n" + json.dumps(summary, indent=2))
    log.info(f"saved {out_dir}/interpret_{args.split}.json")


def main():
    p = argparse.ArgumentParser(description="TC-MIL interpretability analysis")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/interpretability")
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--participant_only", action="store_true")
    p.add_argument("--temporal", default="gru", choices=["none", "gru", "transformer"])
    p.add_argument("--gru_layers", type=int, default=1)
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
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pos_weight", type=float, default=-1.0,
                   help="BCE pos_weight; <0 = auto (neg/pos), 1.0 = no class weighting.")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n_seeds, args.max_epochs, args.patience = 1, 3, 99
    analyze(args)


if __name__ == "__main__":
    main()
