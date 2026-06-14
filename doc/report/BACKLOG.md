# Backlog — Parked (ensembles) + Roadmap (single-model F1)

Decision (2026-06-13): the headline is the **single model**, for a fair
comparison with the text-only literature (which reports single-model per-seed
mean ± std). Ensembling is parked here as an *optional later boost*, not a
headline. All ensemble numbers below are real and leakage-free; they are
removed from the main docs only to keep the focus on the single model.

## Why single-model first (the diagnosis)

The single model **ranks excellently but decides sub-optimally**:

| | value |
| --- | --- |
| Official test AUC (30-seed, single) | **0.856 ± 0.011** |
| Official test macro-F1 (30-seed, single) | 0.751 ± 0.032 |
| Official test micro-F1 (30-seed, single) | 0.794 ± 0.026 |

AUC 0.856 means the probabilities separate the classes well; at the *optimal*
operating point macro-F1 would be ≈0.82+. The ~0.07 gap is **threshold /
calibration**, not ranking. So the highest-ROI work is making the decision
rule extract the F1 the ranking already supports — without ensembling, and
without breaking the leakage-free protocol.

## Single-model F1 roadmap — COMPLETE (2026-06-13)

All 5 levers explored one-at-a-time (`results/full189/SINGLE_MODEL_IMPROVEMENTS.md`).
**Lever 3 (`pos_weight = 1.0`) was the sole robust win**; the rest gave no gain
at 30 seeds. Final single-model recipe = **bge-large + GRU + pos_weight=1.0**,
official 30-seed **AUC 0.864 / macro-F1 0.774 / micro-F1 0.814** — significant
over every legacy baseline and over Milintsevich 0.739 (all p<1e-7,
`results/ablation_stats.md`).

1. **Leakage-free threshold (OOF probe)** — tested, ≤+0.004 macro-F1 vs the
   a-priori prevalence threshold. Per-seed threshold already near the honest
   ceiling. No robust gain.
2. **Probability calibration (temperature scaling)** — monotonic scaling does
   not move the oracle F1; ≈ threshold selection, so no gain beyond lever 1.
3. **Class-weight / loss — `pos_weight = 1.0`. THE WIN.** Dropping the
   recall-biasing ≈2.5 class weight lifted macro-F1 0.751 → 0.774 (sig).
4. **Symptom-informed decision** — no gain; dev (33 subj) too small to tune the
   blend weight without leakage.
5. **Richer single representation (frozen)** — gte-Qwen2-1.5B (1.5B, last-pool)
   *worse* than bge-large on every metric (AUC 0.797 / macro 0.683); rejected.
   The only remaining AUC-ceiling lever is **encoder fine-tuning** (cluster
   suite, `README_FINETUNE.md`) — out of scope for the frozen pipeline.

Constraint kept throughout: train-only fit, dev/OOF selection, test once,
per-seed mean ± std → literature-comparable.

## Parked ensemble results (add back later as optional boosting)

> **BEST PAIR (2026-06-14, roadmap executed) — bge-GRU + UAE-GRU.** Leakage-free
> best-pair search over {bge,mxbai,uae}×{plain,gru} (`training/ensemble_pair_search.py`:
> per-member temperature scaling on dev, logit/arith mean, prevalence threshold,
> select best DISTINCT-encoder pair by dev macro-F1; 4 missing members trained by
> `run_ensemble_roadmap.sh` → `results/full189/ens_members/`). Winner = **bge-large
> GRU + UAE-large GRU**, arithmetic mean (dev macro-F1 0.821). Both members GRU,
> two distinct encoders → fair vs 2-branch fusions.
>
> | Model | micro-F1 | macro-F1 | AUC |
> | --- | --- | --- | --- |
> | bge-GRU + UAE-GRU, full 30-seed avg | 0.838 ± 0.025 | 0.803 ± 0.031 | 0.864 ± 0.008 |
> | **bge-GRU + UAE-GRU, 6×(5+5) indep** | **0.862 ± 0.011** | **0.833 ± 0.015** | **0.868 ± 0.004** |
>
> One-sample t (macro-F1): vs Milintsevich 0.739 p=7.0e-12 (30-seed) / 3.0e-5
> (6×5+5); vs single bar 0.774 Δ=+0.029 p=2.8e-5 → clears. This is the new best
> ensemble: 6×(5+5) macro 0.833 > the bge-plain+mxbai-GRU pair (0.809) and >
> Agarwal's trained single (0.80). Full dev ranking of all 30 pair×rule configs in
> `results/full189/ens_members/pair_search.txt` (GRU pairs dominate, consistent
> with the design ablation). Final single-model recipe unchanged; ensemble is the
> optional boost.

> **RESOLVED — pw1.0 ensemble rebuilt and measured (2026-06-14).** Member B
> (`results/full189/official_mxbai_gru_pw1_30seed`, mxbai GRU, `pos_weight=1.0`,
> 30 seeds base_seed 100, `--eval_test`) now exists and is seed-aligned with
> member A (`single_pw1_30seed`, bge-large plain pw1.0). Combined offline with
> `training/combine_pw1_ensemble.py` (prob-averaging at the a-priori prevalence
> threshold, no retrain). The pw1.0 ensemble **clears the new single bar 0.774
> beyond seed noise** → it earns a headline.

### Official test (full 189, **pos_weight=1.0 members**, 2026-06-14)

| Model | micro-F1 | macro-F1 | AUC |
| --- | --- | --- | --- |
| 2-member (bge plain + mxbai GRU), full 30-seed avg | 0.839 ± 0.023 | 0.804 ± 0.028 | 0.867 ± 0.011 |
| 2-member, 6×(5+5) indep | 0.844 ± 0.016 | 0.809 ± 0.019 | 0.867 ± 0.006 |

One-sample t (macro-F1): vs Milintsevich 0.739 **p = 3.3e-13** (30-seed),
**p = 4.5e-4** (6×5+5); vs the **new single bar 0.774** Δ = **+0.030**,
**p = 3.4e-6** → clears. The pw1.0 ensemble adds +0.030 macro over the single
model (vs only +0.018 for the old auto-member ensemble) *and* still collapses
AUC seed variance (std 0.011 → 0.006). Final single recipe stays bge-large +
GRU + pos_weight=1.0; ensemble is the optional boost on top.

Numbers below are the older **pos_weight=auto** ensembles (archival reference):

### Official test (full 189, pos_weight=auto members)

| Model | micro-F1 | macro-F1 | AUC |
| --- | --- | --- | --- |
| 2-member (bge-large plain + mxbai GRU), 6×(5+5) indep | 0.830 ± 0.021 | 0.792 ± 0.026 | 0.865 ± 0.001 |
| 2-member, 3×(10+10) indep | 0.837 ± 0.020 | 0.801 ± 0.024 | 0.867 ± 0.002 |
| headline point ensemble (locked, OOF threshold) | — | 0.811 | 0.875 |

One-sample t vs Milintsevich macro-F1 0.739: 2-member 6×(5+5) **p = 0.006**.

### Cross-validation (clean pool, seed/member ensembles)

| Protocol | AUC | macro-F1 | F1 |
| --- | --- | --- | --- |
| K-Fold seed-ensemble (GRU, testprev) | 0.827 | — | 0.623 |
| MC 3-member seed-ensemble (s10, testprev) | 0.883 | 0.804 | 0.727 |

### Mechanisms parked

- Seed-ensembling (prob averaging over N seeds of one architecture).
- Multi-encoder ensembling (bge-large + mxbai/UAE; `ensemble_tcmil_official.py`,
  `combine_oof_ensemble.py`).
- OOF-mixed plain+GRU member selection by OOF AUC.
- Pooled-189 protocol (leaky; archival only).

Code remains in place (`ensemble_tcmil_official.py`, `combine_oof_ensemble.py`,
`cv_tcmil.py --member`); only the *reporting* is parked.
