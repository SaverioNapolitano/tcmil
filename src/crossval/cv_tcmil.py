"""TC-MIL cross-validation: Stratified Group K-Fold and Monte Carlo.

Complements the official-protocol script (train_tcmil_official.py) with the
repo's two CV protocols, sharing the same chunking, model, and training
loop.

Leakage rules:
    - Default CV pool is train+dev (142 subjects). The official test split
      stays out of the pool so the official-protocol evaluation remains
      untouched; pass --include_test only to reproduce comparisons against
      the older pooled-189 numbers.
    - Inside each fold, an inner subject-level stratified split (15%)
      provides early stopping and threshold tuning; the fold's test subjects
      are never seen during training, model selection, or thresholding.
    - The encoder is frozen, so chunk embeddings are precomputed once.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.tcmil_data import embed_chunks, load_official_split
from src.crossval.oof_threshold_official import build_oof_probs
from src.training.train_tcmil_official import (
    ChunkBagDataset,
    collate,
    predict,
    set_seed,
    train_one_seed,
    tune_threshold,
)
from src.core.utils.evaluation import (
    run_monte_carlo_cv_ensemble,
    run_stratified_group_k_fold_ensemble,
)
from src.core.utils.metrics import compute_metrics
from src.core.utils.stats import format_aggregate_report


def main():
    p = argparse.ArgumentParser(description="TC-MIL cross-validation")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/tcmil_cv")
    p.add_argument("--mode", choices=["kfold", "mc", "loso"], default="kfold")
    p.add_argument("--include_test", action="store_true",
                   help="Add the official test split to the CV pool (pooled-189 protocol; "
                        "incomparable with the official-split evaluation).")

    p.add_argument("--encoder_name", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--doc_prefix", default="",
                   help='Prefix prepended to every chunk before encoding (e5: "query: ").')
    p.add_argument("--temporal", default="none", choices=["none", "gru", "transformer"],
                   help="Optional context layer over the chunk sequence before MIL pooling.")
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--pos_weight", type=float, default=-1.0,
                   help="Override BCE pos_weight; <0 = auto (neg/pos). 1.0 = no class weighting.")
    p.add_argument("--pooling", default="mean", choices=["mean", "last"],
                   help="Token pooling for the encoder (last = causal-LM embedders).")
    p.add_argument("--aux_mode", default="binary", choices=["binary", "score"],
                   help="Aux target: binarized PHQ-8 items (BCE) or 0-3 subscores (MSE).")
    p.add_argument("--n_repeats", type=int, default=1,
                   help="Repeat K-Fold with shifted fold seeds and aggregate over "
                        "all repeats x folds (kfold mode only).")
    p.add_argument("--member", action="append",
                   help="Ensemble member 'encoder[:temporal]'. Repeatable. When given, "
                        "each run trains one model per member and averages their "
                        "probabilities (overrides --encoder_name/--temporal).")
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

    p.add_argument("--n_folds", type=int, default=5)          # kfold mode
    p.add_argument("--n_splits", type=int, default=5)          # mc mode
    p.add_argument("--test_size", type=float, default=0.2)     # mc mode
    p.add_argument("--val_size", type=float, default=0.15)     # inner split
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold_metric", default="f1",
                   choices=["f1", "bacc", "youden", "prevalence"],
                   help="Per-fold threshold strategy. f1 gives the best CV F1; "
                        "prevalence is more conservative (higher precision).")
    p.add_argument("--threshold_mode", default="innerval",
                   choices=["innerval", "oof", "testprev"],
                   help="innerval: threshold from the inner-val split (default). "
                        "oof: nested OOF over the fold's training pool. NOTE — "
                        "oof helps the OFFICIAL protocol (probe/final train sizes "
                        "match) but REGRESSES CV F1: the nested probe models train "
                        "on fewer subjects than the final per-run model, so their "
                        "probability scale is miscalibrated and the transferred "
                        "threshold sits too high (recall collapses). Kept for "
                        "experimentation only. "
                        "testprev: transductive prevalence — pick t so the predicted "
                        "positive rate ON THE FOLD'S TEST PROBS matches the training "
                        "prevalence. Uses no test labels, only unlabeled test scores, "
                        "and is immune to probability-scale mismatch because the "
                        "threshold is set on the same distribution it is applied to.")
    p.add_argument("--oof_folds", type=int, default=4)   # nested OOF folds
    p.add_argument("--oof_seeds", type=int, default=1)   # seeds per nested fold
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

    # --- CV pool ---
    splits = ["train", "dev"] + (["test"] if args.include_test else [])
    pool = []
    for s in splits:
        pool.extend(load_official_split(args.data_dir, s, args.window, args.stride,
                                        participant_only=args.participant_only))
    log.info(f"CV pool: {len(pool)} subjects from {splits}")

    cache_tag = f"_w{args.window}_s{args.stride}_l{args.max_len}"
    if args.participant_only:
        cache_tag += "_ponly"
    if args.doc_prefix:
        cache_tag += "_pfx" + "".join(c for c in args.doc_prefix if c.isalnum())

    # Ensemble members: each run trains one model per member on its own
    # frozen embeddings; probabilities are averaged before thresholding.
    member_specs = args.member or [f"{args.encoder_name}:{args.temporal}"]
    members = []
    if args.pooling != "mean":
        cache_tag += f"_pool{args.pooling}"
    for spec in member_specs:
        enc, _, temp = spec.partition(":")
        temp = temp or "none"
        mp = [dict(iv) for iv in pool]
        embed_chunks(mp, enc, device, max_len=args.max_len,
                     cache_tag=cache_tag, prefix=args.doc_prefix, pooling=args.pooling)
        members.append({
            "name": f"{enc}:{temp}",
            "temporal": temp,
            "by_id": {iv["interview_id"]: iv for iv in mp},
            "dim": mp[0]["embeddings"].size(1),
        })
    log.info(f"members: {[m['name'] for m in members]}")
    if args.threshold_mode == "oof":
        assert len(members) == 1, "oof threshold mode supports a single member only"
    embedding_dim = members[0]["dim"]

    # Cache the OOF threshold per outer fold so the nested probe runs once,
    # not once per seed-ensemble member (the members are averaged, so they
    # must share one threshold anyway). Keyed by the fold component of the seed.
    oof_t_cache: dict[int, float] = {}

    # --- Per-fold callback: inner val split -> train -> dev-style selection ---
    def train_eval_fn(train_pool, test_set, run_seed):
        set_seed(run_seed)

        sids = sorted(iv["interview_id"] for iv in train_pool)
        sid_labels = {iv["interview_id"]: iv["label"] for iv in train_pool}
        labels = [sid_labels[s] for s in sids]
        # Fold-deterministic inner split: the seed-ensemble aggregation stacks
        # val probabilities across seeds, so every seed within a fold must see
        # the same val subjects. run_seed = base + fold_idx*100 + seed_idx, so
        # flooring to the fold component keeps the split fixed per fold while
        # model init/shuffling still vary by seed.
        fold_seed = run_seed - (run_seed % 100)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=fold_seed)
        tr_idx, va_idx = next(sss.split(np.zeros(len(sids)), labels))
        tr_sids = {sids[i] for i in tr_idx}
        va_sids = {sids[i] for i in va_idx}

        test_sids = {iv["interview_id"] for iv in test_set}
        assert tr_sids.isdisjoint(test_sids) and va_sids.isdisjoint(test_sids), \
            "LEAKAGE: fold test subjects inside training pool"

        pool_prev = float(np.mean([iv["label"] for iv in train_pool]))

        # One model per ensemble member; probabilities averaged across members.
        val_prob_runs, test_prob_runs = [], []
        val_labels = test_labels = None
        for m in members:
            train_ivs = [m["by_id"][s] for s in sorted(tr_sids)]
            val_ivs = [m["by_id"][s] for s in sorted(va_sids)]
            test_ivs = [m["by_id"][iv["interview_id"]] for iv in test_set]
            margs = argparse.Namespace(**{**vars(args), "temporal": m["temporal"]})

            model, _ = train_one_seed(margs, run_seed, train_ivs, val_ivs, device, m["dim"])

            val_loader = DataLoader(ChunkBagDataset(val_ivs), batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate)
            test_loader = DataLoader(ChunkBagDataset(test_ivs), batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate)
            vp, val_labels = predict(model, val_loader, device)
            tp, test_labels = predict(model, test_loader, device)
            val_prob_runs.append(vp)
            test_prob_runs.append(tp)

        val_probs = np.mean(val_prob_runs, axis=0)
        test_probs = np.mean(test_prob_runs, axis=0)

        if args.threshold_mode == "oof":
            # Nested OOF over the whole outer-train pool -> stable threshold.
            if fold_seed not in oof_t_cache:
                m_train_pool = [members[0]["by_id"][iv["interview_id"]] for iv in train_pool]
                oof_probs, oof_labels = build_oof_probs(
                    args, m_train_pool, device, embedding_dim, log)
                oof_auc = compute_metrics(
                    oof_labels, (oof_probs >= 0.5).astype(int), oof_probs)["roc_auc"]
                oof_t_cache[fold_seed] = tune_threshold(
                    oof_labels, oof_probs,
                    metric=args.threshold_metric, prevalence=pool_prev)
                log.info(f"      [fold {fold_seed}] OOF N={len(oof_labels)} "
                         f"AUC={oof_auc:.4f} t={oof_t_cache[fold_seed]:.2f}")
            best_t = oof_t_cache[fold_seed]
        elif args.threshold_mode == "testprev":
            best_t = tune_threshold(None, test_probs,
                                    metric="prevalence", prevalence=pool_prev)
        else:
            best_t = tune_threshold(val_labels, val_probs,
                                    metric=args.threshold_metric, prevalence=pool_prev)
        metrics = compute_metrics(test_labels, (test_probs >= best_t).astype(int), test_probs)
        metrics["probability"] = test_probs.tolist()
        metrics["true_label"] = test_labels.tolist()
        metrics["val_probability"] = val_probs.tolist()
        metrics["val_true_label"] = val_labels.tolist()
        return metrics

    # --- Ensemble threshold: match the per-run strategy ---
    pool_prev = float(np.mean([iv["label"] for iv in pool]))

    if args.threshold_mode == "testprev":
        def ensemble_threshold_fn(group_idx, val_y, avg_val_probs, avg_test_probs):
            return tune_threshold(None, avg_test_probs,
                                  metric="prevalence", prevalence=pool_prev)
    elif args.threshold_mode == "oof":
        def ensemble_threshold_fn(group_idx, val_y, avg_val_probs, avg_test_probs):
            fold_seed = ((args.seed + group_idx * 100) // 100) * 100
            return oof_t_cache.get(fold_seed, 0.5)
    else:
        ensemble_threshold_fn = None  # default: loss-tuned on ensembled inner-val

    # --- Run CV ---
    if args.mode == "kfold":
        assert not (args.n_repeats > 1 and args.threshold_mode == "oof"), \
            "oof threshold cache is not repeat-aware"
        all_raw, ens_metrics = [], []
        for rep in range(args.n_repeats):
            rep_seed = args.seed + 1000 * rep
            _, raw, _ = run_stratified_group_k_fold_ensemble(
                pool, train_eval_fn,
                n_folds=args.n_folds, n_seeds_per_fold=args.n_seeds,
                random_state=rep_seed,
                ensemble_threshold_fn=ensemble_threshold_fn,
            )
            for m in raw:
                m["_repeat"] = rep
            all_raw.extend(raw)
            # Re-derive per-fold seed ensembles from the raw predictions so
            # repeats can be pooled into one aggregate.
            groups: dict[int, list[dict]] = {}
            for m in raw:
                if "probability" not in m:
                    continue
                g = rep * 1000 + m["_fold_idx"]
                groups.setdefault(g, []).append({
                    k: np.array(m[k]) for k in
                    ("probability", "true_label", "val_probability", "val_true_label")
                    if k in m
                })
            from src.core.utils.evaluation import _seed_ensemble_metrics
            ens_metrics.extend(_seed_ensemble_metrics(
                groups, "_fold_idx", ensemble_threshold_fn))
        from src.core.utils.stats import compute_aggregate_metrics
        per_run_agg = compute_aggregate_metrics(all_raw)
        ens_agg = compute_aggregate_metrics(ens_metrics)
        raw = all_raw
        ens_label = (f"Seed-Ensemble (per fold, {args.n_repeats} repeat(s) x "
                     f"{args.n_folds} folds)")
    elif args.mode == "mc":
        per_run_agg, raw, ens_agg = run_monte_carlo_cv_ensemble(
            pool, train_eval_fn,
            n_splits=args.n_splits, n_seeds_per_split=args.n_seeds,
            test_size=args.test_size, random_state=args.seed,
            ensemble_threshold_fn=ensemble_threshold_fn,
        )
        ens_label = "Seed-Ensemble (per split)"
    else:  # loso: pooled OOF predictions over the whole pool
        from src.core.utils.evaluation import run_leave_one_subject_out_cv
        boot_agg, raw = run_leave_one_subject_out_cv(
            pool, train_eval_fn,
            n_seeds_per_fold=args.n_seeds, random_state=args.seed,
        )
        # Pooled metrics at the prevalence-matched threshold (rate matching
        # only; no per-label optimization on the pooled OOF probs).
        y_true = np.array([m["true_label"][0] for m in raw if m.get("true_label")])
        y_prob = np.array([m["probability"][0] for m in raw if m.get("probability")])
        t = tune_threshold(None, y_prob, metric="prevalence", prevalence=pool_prev)
        pooled = compute_metrics(y_true, (y_prob >= t).astype(int), y_prob)
        per_run_agg = boot_agg
        ens_agg = {k: {"mean": v, "std": 0.0, "ci_lower": v, "ci_upper": v}
                   for k, v in pooled.items()}
        ens_agg["_threshold"] = {"mean": t, "std": 0.0, "ci_lower": t, "ci_upper": t}
        ens_label = f"LOSO pooled (prevalence t={t:.2f})"

    report = (f"## Per-Run\n{format_aggregate_report(per_run_agg)}\n\n"
              f"## {ens_label}\n{format_aggregate_report(ens_agg)}\n")
    payload = {"per_run_aggregate": per_run_agg, "ensemble_aggregate": ens_agg, "raw": raw}

    with open(out_dir / "cv_report.txt", "w") as f:
        f.write(f"# TC-MIL {args.mode} CV (pool={'+'.join(splits)})\n\n{report}")
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info("\n" + report)
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
