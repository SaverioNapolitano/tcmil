"""Zero-shot transfer: a DAIC-WOZ-trained TC-MIL model evaluated on E-DAIC.

Loads a trained checkpoint directory (``config.json`` + per-seed ``seed_*.pt``
written by ``train_tcmil_official.py``), rebuilds the seed ensemble, and scores
the E-DAIC (AVEC 2019) splits without any further training. E-DAIC is
un-diarized participant-only ASR, so instances are built with silence-gap
segmentation; pass the SAME ``--gap_merge``/``--window``/``--stride`` used to
train the source model so the only shift measured is the domain shift.

The encoder, dims and temporal mode are read from the checkpoint's
``config.json``. The frozen encoder embeds E-DAIC chunks into a *separate*
cache namespace (``_edaic_gap*``) so E-DAIC and DAIC-WOZ — which overlap in
participant-ID ranges — never share cached embeddings.

Reporting (all on the ensemble-mean probability):
    - AUC / PR-AUC: threshold-free, the primary zero-shot numbers.
    - F1 @ 0.5.
    - Threshold tuned on the E-DAIC *dev* split (per --threshold_metric),
      applied to E-DAIC test, plus every strategy for the trade-off table.
    - Optionally a fixed --source_threshold (e.g. the DAIC dev threshold) to
      report the strictest no-target-tuning transfer number.

Usage
-----
    python -m src.training.eval_edaic_zeroshot \
        --checkpoint_dir results/single_model/headline_pw1_30seed_ckpt/checkpoints \
        --edaic_dir data/e-daic --gap_merge 2.0 \
        --output_dir results/zeroshot/edaic_headline
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

from src.core.models.tcmil import TCMIL
from src.core.tcmil_data import embed_chunks, load_edaic_split
from src.core.utils.metrics import compute_metrics
from src.training.train_tcmil_official import (
    ChunkBagDataset,
    collate,
    predict,
    predict_symptom_sum,
    tune_threshold,
)


def main():
    p = argparse.ArgumentParser(description="E-DAIC zero-shot evaluation of a DAIC-WOZ TC-MIL model")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Dir with config.json and seed_*.pt (from train_tcmil_official).")
    p.add_argument("--edaic_dir", default="data/e-daic",
                   help="E-DAIC dataset root (labels/ + raw/).")
    p.add_argument("--output_dir", default="results/zeroshot/edaic")
    p.add_argument("--gap_merge", type=float, default=2.0,
                   help="Silence-gap (s) segmentation; match the source model's training value.")
    p.add_argument("--max_len", type=int, default=256,
                   help="Encoder max tokens (config.json does not store it; match training).")
    p.add_argument("--doc_prefix", default="",
                   help='Per-chunk encoder prefix (e5: "query: "); match training.')
    p.add_argument("--pooling", default="mean", choices=["mean", "last"],
                   help="Token pooling; match training.")
    p.add_argument("--threshold_metric", default="f1",
                   choices=["f1", "bacc", "youden", "prevalence"],
                   help="Strategy for the dev-tuned headline threshold.")
    p.add_argument("--source_threshold", type=float, default=None,
                   help="Optional fixed threshold (e.g. DAIC dev) for no-target-tuning transfer.")
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    cfg = json.load(open(ckpt_dir / "config.json"))
    log.info(f"checkpoint config: {cfg}")
    if "embedding_dim" not in cfg:
        raise ValueError(
            f"{ckpt_dir}/config.json is not a single-encoder TC-MIL checkpoint "
            "(no embedding_dim). Ensemble/NCL checkpoints are not supported here."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    window = cfg["window"]
    stride = cfg["stride"]
    encoder_name = cfg["encoder_name"]

    # --- Build E-DAIC bags (participant-only, gap-merged) ---
    dev_ivs = load_edaic_split(args.edaic_dir, "dev", window, stride, gap_merge=args.gap_merge)
    test_ivs = load_edaic_split(args.edaic_dir, "test", window, stride, gap_merge=args.gap_merge)
    log.info(f"edaic dev={len(dev_ivs)} test={len(test_ivs)}")

    # Distinct cache namespace: E-DAIC shares PID ranges with DAIC-WOZ, and the
    # gap-merge construction differs from the diarized one, so both go into the
    # tag to guarantee no cross-dataset / cross-construction cache reuse.
    gap_s = str(args.gap_merge).replace(".", "p")
    cache_tag = f"_w{window}_s{stride}_l{args.max_len}_edaic_gap{gap_s}"
    if args.doc_prefix:
        cache_tag += "_pfx" + "".join(c for c in args.doc_prefix if c.isalnum())
    if args.pooling != "mean":
        cache_tag += f"_pool{args.pooling}"
    for ivs in (dev_ivs, test_ivs):
        embed_chunks(ivs, encoder_name, device, max_len=args.max_len,
                     cache_tag=cache_tag, prefix=args.doc_prefix, pooling=args.pooling)
    embedding_dim = dev_ivs[0]["embeddings"].size(1)
    if embedding_dim != cfg["embedding_dim"]:
        raise ValueError(
            f"embedding_dim mismatch: encoder gives {embedding_dim}, "
            f"checkpoint expects {cfg['embedding_dim']}."
        )

    dev_loader = DataLoader(ChunkBagDataset(dev_ivs), batch_size=16,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ChunkBagDataset(test_ivs), batch_size=16,
                             shuffle=False, collate_fn=collate)

    # --- Load every seed head, predict, seed-ensemble (mean prob) ---
    seed_paths = sorted(ckpt_dir.glob("seed_*.pt"))
    if not seed_paths:
        raise FileNotFoundError(f"No seed_*.pt in {ckpt_dir}")
    log.info(f"ensembling {len(seed_paths)} seed checkpoints")

    dev_prob_runs, test_prob_runs = [], []
    dev_sym_runs, test_sym_runs = [], []
    dev_labels = test_labels = None
    for sp in seed_paths:
        model = TCMIL(
            embedding_dim=embedding_dim, proj_dim=cfg["proj_dim"],
            attn_dim=cfg["attn_dim"], dropout=cfg["dropout"],
            temporal=cfg.get("temporal", "none"), gru_layers=cfg.get("gru_layers", 1),
        ).to(device)
        model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))

        dp, dev_labels = predict(model, dev_loader, device)
        tp, test_labels = predict(model, test_loader, device)
        dev_prob_runs.append(dp)
        test_prob_runs.append(tp)
        dev_sym_runs.append(predict_symptom_sum(model, dev_loader, device))
        test_sym_runs.append(predict_symptom_sum(model, test_loader, device))

    dev_avg = np.mean(dev_prob_runs, axis=0)
    test_avg = np.mean(test_prob_runs, axis=0)
    dev_prev = float(np.mean(dev_labels))

    # Threshold tuned on E-DAIC dev (target-dev tuning) for each strategy.
    strategies = {
        "f1": tune_threshold(dev_labels, dev_avg, "f1"),
        "bacc": tune_threshold(dev_labels, dev_avg, "bacc"),
        "youden": tune_threshold(dev_labels, dev_avg, "youden"),
        "prevalence": tune_threshold(dev_labels, dev_avg, "prevalence", prevalence=dev_prev),
    }
    best_t = strategies[args.threshold_metric]

    dev_metrics = compute_metrics(dev_labels, (dev_avg >= best_t).astype(int), dev_avg)
    log.info(f"EDAIC-DEV [{args.threshold_metric}] (t={best_t:.2f}): " +
             " ".join(f"{k}={v:.4f}" for k, v in dev_metrics.items() if isinstance(v, float)))

    # Test: threshold-free + 0.5 + every dev-tuned strategy.
    test_auc = compute_metrics(test_labels, (test_avg >= 0.5).astype(int), test_avg)
    log.info(f"EDAIC-TEST threshold-free: AUC={test_auc['roc_auc']:.4f} "
             f"PR-AUC={test_auc['pr_auc']:.4f} | F1@0.5={test_auc['f1']:.4f}")

    test_by_strategy = {}
    for name, t in strategies.items():
        tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
        test_by_strategy[name] = {"threshold": t, **tm}
        log.info(f"EDAIC-TEST [{name}] (t={t:.2f}): "
                 f"F1={tm['f1']:.4f} P={tm['precision']:.4f} R={tm['recall']:.4f} "
                 f"BAcc={tm['balanced_accuracy']:.4f} macroF1={tm['macro_f1']:.4f}")

    # Per-seed test spread (literature-comparable: mean +- std, dev-tuned t).
    per_seed_test = [
        compute_metrics(test_labels, (pr >= best_t).astype(int), pr)
        for pr in test_prob_runs
    ]
    for k in ("precision", "recall", "f1", "macro_f1", "roc_auc"):
        vals = [m[k] for m in per_seed_test]
        log.info(f"EDAIC-TEST per-seed {k} (t={best_t:.2f}): "
                 f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results = {
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_config": cfg,
        "eval_args": vars(args),
        "n_seeds": len(seed_paths),
        "edaic_dev_n": len(dev_ivs),
        "edaic_test_n": len(test_ivs),
        "edaic_dev_prevalence": dev_prev,
        "edaic_test_prevalence": float(np.mean(test_labels)),
        "thresholds": strategies,
        "dev_threshold": best_t,
        "dev_ensemble": dev_metrics,
        "test_threshold_free": {"roc_auc": test_auc["roc_auc"], "pr_auc": test_auc["pr_auc"]},
        "test_f1_at_0.5": test_auc["f1"],
        "test_by_strategy": test_by_strategy,
        "test_ensemble": test_by_strategy[args.threshold_metric],
        "test_per_seed": per_seed_test,
        "dev_probs": dev_avg.tolist(),
        "dev_labels": np.asarray(dev_labels).tolist(),
        "test_probs": test_avg.tolist(),
        "test_labels": np.asarray(test_labels).tolist(),
        "dev_prob_runs": [pr.tolist() for pr in dev_prob_runs],
        "test_prob_runs": [pr.tolist() for pr in test_prob_runs],
        "dev_sym_runs": [s.tolist() for s in dev_sym_runs],
        "test_sym_runs": [s.tolist() for s in test_sym_runs],
    }

    if args.source_threshold is not None:
        st = float(args.source_threshold)
        sm = compute_metrics(test_labels, (test_avg >= st).astype(int), test_avg)
        results["test_source_threshold"] = {"threshold": st, **sm}
        log.info(f"EDAIC-TEST [source t={st:.2f}] (no target tuning): "
                 f"F1={sm['f1']:.4f} P={sm['precision']:.4f} R={sm['recall']:.4f} "
                 f"BAcc={sm['balanced_accuracy']:.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
