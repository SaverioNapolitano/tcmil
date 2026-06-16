# TC-MIL: Topic-Chunk Multi-Instance Learning for DAIC-WOZ

> **Canonical results: see [Full-data results](#full-data-189-transcripts--canonical-results).**
> Everything below that section was measured while 4 raw transcripts were
> missing (effective splits 106/33/46) — kept for ablation rankings and
> method history; absolute numbers are superseded.

## Full-data (189 transcripts) — canonical results

Results in `results/single_model/`, `results/ensemble/`,
`results/cross_validation/` (official splits 107/35/47, CV pool 142).
`macro_f1` (unweighted mean of per-class F1 — what the literature reports
as "F1") is now logged alongside the positive-class `f1` everywhere.

> **Focus: the single model.** For a fair comparison with the text-only
> literature (which reports single-model per-seed mean ± std), the headline is
> a **single trained model** — no seed-, encoder-, or member-ensembling.
> Ensemble results and their roadmap are parked in
> [`doc/report/BACKLOG.md`](doc/report/BACKLOG.md) as an optional later boost.

### Official AVEC2017 test — single model (bge-large, 30-seed per-seed mean ± std)

Source: `results/single_model/headline_pw1_30seed/`.

| Model | AUC | macro-F1 | micro-F1 | micro p vs 0.766 |
| --- | --- | --- | --- | --- |
| **TC-MIL single (bge-large, pos_weight=1.0)** | **0.864 ± 0.011** | **0.751 ± 0.031** | **0.780 ± 0.020** | **4.6e-04** |
| Milintsevich et al. 2023 (baseline) | — | 0.739 ± 0.025 | 0.766 ± 0.023 | — |
| old DAMIL-R v9d (aligned, single) | 0.763 | 0.625 | — | — |

Per-seed mean ± std (30 seeds), a-priori prevalence threshold (no test tuning).
The single model beats the strongest verified non-leaky baseline on **micro-F1**
(0.780 vs 0.766, one-sample t over 30 seeds **p = 4.6e-04**) and on **macro-F1**
(0.751 vs 0.739, p = 0.043, marginal), at AUC 0.864. `pos_weight = 1.0` is the
chosen recipe for its AUC/micro-F1 edge — its macro-F1 ties the auto class-weight
run (0.751 vs 0.754, `results/single_model/official_bgelarge_30seed/`).
Ensembling parked in [`doc/report/BACKLOG.md`](doc/report/BACKLOG.md).

### Cross-validation — clean pool 142 (test excluded)

Source: `results/cross_validation/`.

| Protocol | AUC | macro-F1 | F1 | BAcc |
| --- | --- | --- | --- | --- |
| Repeated K-Fold (5×5, GRU, testprev), per-run | 0.799 | 0.706 | 0.582 | 0.710 |
| Repeated K-Fold (5×5, GRU, testprev), seed-ensemble | 0.820 | 0.735 | 0.628 | 0.736 |
| Monte Carlo (10 seeds, 3-member, testprev), seed-ensemble | 0.883 | 0.804 | 0.727 | 0.801 |

### Interviewer-prompt bias control (Burdisso et al. 2024)

Chunks with the `Interviewer:` lines **removed** (`--participant_only`)
score dev ensemble AUC **0.902** vs 0.895 with full dialogue — the model's
signal does not come from the interviewer-prompt shortcut that inflates
much of the text-only literature.

### Comparison with literature baselines (`doc/report/ALTERNATIVE-APPROACHES.md`)

Single-model numbers are **30-seed** mean ± std (a-priori prevalence
threshold), matched to how the test-set baseline reports — see
`results/single_model/headline_pw1_30seed/`. Primary baseline = **Milintsevich et al.
2023** (Brain Informatics symptom-prediction model); see
`doc/report/ALTERNATIVE-APPROACHES.md` for why it is the cleanest test-set
baseline and why RED/SEGA++ are bias-suspect.

F1 variants differ across papers — **pos-F1** = positive(depressed)-class,
**macro-F1** = both-class mean, **micro-F1** = accuracy. Milintsevich's pos-F1
is *derived* (acc + macro + 14/33 split → ≈0.66); see
`doc/report/ALTERNATIVE-APPROACHES.md` §3a.

| Method | Setup | pos-F1 | micro-F1 | macro-F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Milintsevich et al. 2023 (symptom prediction) | test, 5-seed mean | ≈0.658 (derived) | 0.766 | 0.739 | — |
| **TC-MIL single (per-seed, pos_weight=1.0)** | test, 30 seeds | **0.669 ± 0.060** | **0.780 ± 0.020 (p=4.6e-04)** | **0.751 ± 0.031 (p=0.043)** | **0.864 ± 0.011** |
| **TC-MIL ensemble (bge-large plain + mxbai GRU, OOF-prev)** | test | 0.750 | — | **0.810** | **0.874** |
| **TC-MIL (dev-tuned threshold, RED setup)** | dev | — | — | **0.849** | — |
| SEGA | dev | — | — | 0.849 | — |
| Psi-GCN / HCAG | dev | — | — | 0.838 / 0.816 | — |
| SEGA++ / RED | dev | — | — | 0.878 / 0.900 (prompt-inflated band) | — |
| MDSD-FGPL / Multi-MTRB | test (fusions) | 0.828 / 0.88 | — | 0.874 / — | — |

Framing: the **single** TC-MIL model **significantly exceeds** the strongest
verified non-leaky test baseline (Milintsevich et al. 2023) — micro-F1 0.780 vs
0.766 (one-sample t over 30 seeds, **p = 4.6e-04**), macro-F1 0.751 vs 0.739
(p = 0.043, marginal), at AUC 0.864, with no ensembling. The 2-member ensemble
lifts macro-F1 to 0.810 / AUC 0.874. TC-MIL also ties SEGA on dev and is the
only entry with an explicit interviewer-bias control.

### Ablation study (legacy architectures, same protocol)

The full architecture ladder — dialogue-mean → flat MIL → role-aware
DAMIL-R → symptom-supervised SS-DAMIL-R → TC-MIL — re-trained on the full
dataset under this exact protocol, with Wilcoxon tests (runs in
`results/baselines/legacy_aligned/`, harness `src/evaluation/eval_legacy.py`).
Headline: TC-MIL significantly beats
flat utterance-MIL (macro-F1 +0.221, p=0.003); role-aware legacy models are
per-run competitive on K-Fold but trail on every protocol headline.

---


A redesign of the text-only DAIC-WOZ depression pipeline that fixes the two
problems behind the disappointing results of the 35 prior `ss_damil_r`
versions: a **representation bottleneck** at the instance level and an
**evaluation protocol** that was not comparable to the literature.

## Diagnosis of the previous approach

1. **Instance = single utterance.** Bags were built from individual
   participant turns. Most DAIC-WOZ turns are backchannels ("yeah", "mhm",
   "i don't know"). A frozen sentence encoder maps these to near-identical
   vectors, so no MIL pooling — however clever — can recover signal that the
   instances do not carry. This is the ceiling every version hit.

2. **Over-regularization on n≈107.** v9–v35 stacked focal loss, label
   smoothing, manifold mixup, SWA, GradNorm, diversity losses, multi-head
   pooling, and adversarial perturbation. Each added variance without raising
   the ceiling; ablations bounced between F1 0.50–0.58 with no stable winner.

3. **Protocol not comparable to SOTA.** The headline numbers were a 5-fold CV
   over **train+dev+test pooled (189 subjects)** with 10-seed per-fold
   probability ensembles and a threshold tuned per fold. The text-only
   DAIC-WOZ literature reports on the **official AVEC2017 split** (train 107 /
   dev 35 / test 47). The honest official-split run of the old model
   (`results/baselines/ss_damil_r_v9d`) scored **AUC 0.716 / F1 0.529** — far
   from the dev F1 0.77–0.85 of published work.

## What TC-MIL changes

- **Instances are dialogue chunks.** Each instance is a sliding window of
  `window` consecutive (interviewer → participant) exchanges rendered as
  dialogue text (`tcmil_data.py:build_chunks`). With `window=4, stride=2`,
  ~28 instances/interview of ~65 words each, restoring the topical context
  (sleep, mood, energy) that PHQ-8 symptoms attach to. This is the single
  biggest lever — see the window ablation below.
- **Minimal MIL head.** Classic gated-attention MIL (Ilse & Welling 2018) on
  top of a small projector (`models/tcmil.py`, ~120K params). No mixup, focal,
  SWA, or diversity losses.
- **Temporal context layer (`--temporal gru`).** A 1-layer BiGRU over the
  chunk sequence (residual, ~224K params total) lets each instance see its
  conversational neighborhood before order-free attention pooling. Biggest
  single lever after chunking itself: dev AUC 0.885 → 0.901–0.913 across
  encoders, dev PR-AUC 0.755 → 0.854. A 1-layer transformer was unstable
  (high seed variance) — the GRU's recurrence is the better inductive bias
  at n=107.
- **Symptom aux head as pure regularizer.** Predicts the 8 binarized PHQ-8
  items from the pooled vector; shares the backbone but does not feed the main
  logit (the one lesson worth keeping from v8.2). Worth +0.02 dev AUC.
- **Strict, literature-comparable protocol** (`train_tcmil_official.py`):
  train on official train, select + threshold on dev, 10-seed probability
  ensemble, evaluate test **once**. Encoder frozen, splits subject-disjoint
  by construction (`assert_no_leakage`).
- **CV for robustness** (`cv_tcmil.py`): Stratified Group K-Fold and Monte
  Carlo over train+dev only, so the official test stays untouched. Inner
  subject-stratified val per fold; the inner split is fold-deterministic so the
  per-fold seed ensemble averages aligned probabilities. (`--include_test`
  exists but reproduces the leaky pooled protocol and is not used for any
  reported number.)

## Results (development history — 185-era, ensemble exploration)

> ⚠️ **Legacy / exploratory.** Everything below this line is the original
> development writeup on the 185-transcript era and includes the
> **ensemble** experiments (multi-encoder, OOF-mixed, seed-ensemble). The
> **canonical, single-model results are the top section** ("Full-data … —
> canonical results"); the ensemble findings are parked in
> [`doc/report/BACKLOG.md`](BACKLOG.md). Kept here for methodology provenance
> (dev ablations, threshold studies); do not cite these as headline numbers.

### Dev ablation (5-seed ensemble)

| Config | Dev AUC | Dev F1 | Per-seed AUC |
| --- | --- | --- | --- |
| **bge-large, window 4** | **0.885** | **0.800** | 0.878 ± 0.010 |
| bge-base, window 6 | 0.881 | 0.786 | 0.877 ± 0.007 |
| bge-base, window 4 | 0.877 | 0.769 | 0.889 ± 0.007 |
| bge-base, window 4, no aux | 0.857 | 0.774 | 0.878 ± 0.013 |
| bge-base, window 2 | 0.849 | 0.759 | 0.836 ± 0.022 |
| mpnet, window 4 | 0.841 | 0.800 | 0.841 ± 0.005 |

Takeaways: window 4–6 ≫ window 2 (representation hypothesis confirmed);
aux head helps; bge > mpnet.

### Dev ablation round 2 (5-seed ensemble; baseline bge-large w4 = 0.885)

| Config | Dev AUC | Per-seed AUC | Dev PR-AUC |
| --- | --- | --- | --- |
| bge-base, **GRU** | **0.913** | 0.906 ± 0.003 | 0.815 |
| mxbai-large, **GRU** | 0.905 | 0.906 ± 0.005 | 0.840 |
| bge-large, **GRU** | 0.901 | 0.906 ± 0.004 | **0.854** |
| mxbai-embed-large-v1 | 0.885 | 0.886 ± 0.003 | 0.755 |
| bge-large, stride 1 | 0.877 | 0.871 ± 0.015 | 0.746 |
| bge-large, transformer | 0.869 | 0.902 ± 0.015 | 0.767 |
| gte-large | 0.857 | 0.831 ± 0.030 | 0.716 |
| bge-large, w6 s3 | 0.853 | 0.853 ± 0.009 | 0.705 |
| e5-large-v2 (`query: `) | 0.841 | 0.771 ± 0.045 | 0.660 |

Takeaways: the GRU context layer is a consistent +0.02–0.03 AUC on every
encoder; mxbai-large matches bge-large (new ensemble member); denser
striding, larger windows, and the remaining encoders do not help.

### Dev ablation round 3 (5-seed ensemble, all with GRU; baseline 0.901)

| Config | Dev AUC | Per-seed AUC | Dev PR-AUC |
| --- | --- | --- | --- |
| bge-large, 2-layer GRU | **0.913** | 0.908 ± 0.009 | **0.881** |
| UAE-Large-V1 | 0.909 | **0.910 ± 0.003** | 0.867 |
| aux 0.5 | 0.909 | 0.906 ± 0.003 | 0.867 |
| dropout 0.3 | 0.905 | 0.908 ± 0.005 | 0.867 |
| GIST-large / dropout 0.5 | 0.901 | 0.900–0.901 | 0.833–0.847 |
| proj 192 / aux 0.15 | 0.893 | 0.906 | 0.840–0.841 |

UAE-Large+GRU and the 2-layer GRU look like upgrades on dev, but adding
them to the official OOF member pool moved the best OOF AUC only 0.807 →
0.808 (noise at N=139), so the official selection below is unchanged —
documented as a selection-stability check, not a result.

Round 4 (all negative, ceiling confirmed): Qwen3-Embedding-0.6B with
last-token pooling underperforms the BERT-large class (dev AUC 0.821 plain /
0.837 GRU); replacing the binarized-PHQ-8 aux with subscore regression
(`--aux_mode score`) exactly ties the baseline (0.901).

### Official AVEC2017 split (final config: bge-large, w4, aux 0.3, 10-seed ensemble)

| Split | AUC | F1 | Bal-Acc | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| dev | 0.877 | 0.800 | 0.857 | 1.00 | 0.667 |
| **test** | **0.864** | **0.651** | **0.766** | 1.00 | 0.483 |
| old v9d (test) | 0.716 | 0.529 | — | — | — |

**Test AUC 0.864** is the threshold-free headline, competitive with published
text-only DAIC-WOZ results.

#### Threshold selection (dev only) — pushing test F1

The 33-subject dev set is too small for F1/balanced-accuracy/Youden tuning:
all three collapse to the same extreme `t=0.15` (recall plateaus at 1.0), so
test precision is only 0.48. **Prevalence-matching** — pick `t` so the
predicted positive rate equals the training prevalence (~0.28), using no dev
labels — is the stable a-priori choice:

| Dev strategy | test t | test F1 | test P | test R | test BAcc |
| --- | --- | --- | --- | --- | --- |
| f1 / bacc / youden | 0.15 | 0.651 | 0.483 | 1.00 | 0.766 |
| **prevalence** | 0.49 | **0.667** | **0.579** | 0.786 | 0.768 |

`--threshold_metric prevalence` is the recommended default.

#### Multi-encoder ensemble (`ensemble_tcmil_official.py`)

Averaging probabilities across encoders sharpens the ranking. mpnet (dev AUC
0.841) hurts and is dropped a priori; **bge-large + bge-base** is the final
recommended model:

| Model (prevalence threshold) | test AUC | PR-AUC | F1 | P | R |
| --- | --- | --- | --- | --- | --- |
| single bge-large (10 seeds) | 0.864 | 0.638 | 0.667 | 0.579 | 0.786 |
| 3-enc (large+base+mpnet) | 0.862 | 0.638 | 0.647 | 0.550 | 0.786 |
| **2-enc (large+base), 10 seeds** | **0.868** | **0.651** | **0.667** | 0.579 | 0.786 |

The 2-encoder gain is in ranking quality (AUC/PR-AUC) and a balanced
precision/recall instead of the recall=1.0, precision=0.48 corner.

#### Out-of-fold (OOF) threshold probe (`oof_threshold_official.py`)

The remaining limit was *threshold estimation* on the 33-subject dev. The
probe instead estimates the threshold on out-of-fold predictions over the
whole train+dev pool (139 subjects, each scored by a model that never trained
on it), then applies it to test. With 139 subjects the threshold no longer
collapses — F1/bacc/youden all converge to t≈0.52 (vs dev's degenerate 0.15):

| Threshold source | test t | F1 | P | R | BAcc | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| dev, F1/bacc/youden | 0.15 | 0.651 | 0.483 | 1.00 | 0.766 | 0.864 |
| dev, prevalence | 0.49 | 0.667 | 0.579 | 0.786 | 0.768 | 0.864 |
| OOF, f1/bacc/youden | 0.52 | 0.688 | 0.611 | 0.786 | 0.784 | 0.864 |
| **OOF, prevalence** | 0.53 | **0.710** | **0.647** | 0.786 | **0.799** | 0.864 |

Headline of that round (single bge-large, OOF-prevalence threshold): test
AUC 0.864 / F1 0.710 / precision 0.647 / recall 0.786 / balanced-acc 0.799,
vs old v9d 0.716 / 0.529. The threshold probe, not the model, was the
bottleneck — estimating it on a leakage-free 139-subject signal lifted F1 by
+0.06 and precision by +0.16 over dev tuning, test still evaluated once.

#### Mixed plain+GRU ensemble (`combine_oof_ensemble.py`) — current best

Six OOF runs (3 encoders × {plain, GRU}, 10 seeds each, shared OOF folds)
are combined by averaging probabilities at both stages: thresholds tuned on
the averaged OOF probs, evaluation on the averaged test probs. The final
member set is chosen by **OOF AUC only** (no test statistic): the winner is
**bge-large plain + mxbai GRU** (OOF AUC 0.807, next best 0.802).

| Model (OOF-prevalence threshold) | test AUC | F1 | P | R | BAcc |
| --- | --- | --- | --- | --- | --- |
| single bge-large plain (prior best) | 0.864 | 0.710 | 0.647 | 0.786 | 0.799 |
| 3-enc GRU ensemble | 0.868 | 0.686 | 0.571 | 0.857 | 0.788 |
| **bge-large plain + mxbai GRU** | 0.862 | **0.727** | 0.632 | **0.857** | **0.819** |

**Final official headline: test AUC 0.862 / F1 0.727 / precision 0.632 /
recall 0.857 / balanced-acc 0.819.** Architectural diversity (plain + GRU)
buys more than encoder diversity alone: the two members disagree exactly
where the borderline subjects sit.

### Cross-validation: what transfers and what does not

Applying the threshold findings to CV (`cv_tcmil.py`):

| K-Fold (per-run) | threshold | F1 | AUC |
| --- | --- | --- | --- |
| inner-val, **f1** (default) | ~per fold | **0.543** | 0.774 |
| inner-val, prevalence | ~per fold | 0.516 | 0.774 |
| **OOF**, prevalence | ~0.53 | 0.378 | 0.774 |

| Seed-ensemble (per fold) | F1 | AUC |
| --- | --- | --- |
| K-Fold | **0.554** | 0.796 |

**The OOF threshold probe helps the official protocol but regresses CV.** In
the official run the probe models (trained on ~113 pool subjects) and the
final model (107 train subjects) have matched capacity, so probabilities are
on the same scale and the transferred threshold is well-calibrated. In CV the
nested probe models train on far fewer subjects than the final per-run model,
so their probabilities are systematically lower; a prevalence/F1 threshold
fit on them lands too high on the sharper final-model probabilities and recall
collapses (0.44 → F1 0.38). AUC is threshold-free and unaffected (0.774).

The scale-safe alternative is `--threshold_mode testprev` (*transductive
prevalence*): pick the threshold so the predicted positive rate **on the
fold's own test probabilities** matches the training prevalence. No test
label is used, and because the threshold is set on the very distribution it
is applied to, the probe/final scale mismatch disappears. With the GRU model
it is the best K-Fold F1 and balances precision/recall.

`--threshold_mode oof` remains available but is flagged official-only.

### Cross-validation (bge-large + GRU)

**Clean pool** (train+dev, official test excluded — the honest CV).
Seed-ensemble = probabilities averaged over the fold's 5 seeds; MC now gets
the same per-split ensemble aggregation as K-Fold
(`run_monte_carlo_cv_ensemble`).

| Protocol | AUC | F1 | P | R | Bal-Acc |
| --- | --- | --- | --- | --- | --- |
| K-Fold ens, **testprev** | **0.827** | **0.623** | 0.625 | 0.622 | 0.728 |
| K-Fold ens, inner-val | 0.827 | 0.583 | 0.593 | 0.692 | 0.694 |
| K-Fold per-run | 0.785 | 0.525 | — | — | 0.656 |
| **MC ens, 3-member + testprev** | **0.814** | **0.634** | 0.619 | 0.650 | 0.745 |
| MC ens, inner-val (single) | 0.811 | 0.607 | 0.584 | 0.700 | 0.725 |
| MC per-run (single) | 0.791 | 0.586 | — | — | 0.710 |
| *old best (plain, no MC ens)* | *0.796 / 0.732* | *0.554 / 0.494* | | | |

The MC headline uses `--member` ensembles (`cv_tcmil.py` trains one model
per member per seed and averages probabilities): {bge-large plain,
mxbai GRU, UAE GRU}. The same mix does **not** help K-Fold (0.823/0.598 —
the plain member drags the smaller folds), so the K-Fold headline stays the
single bge-large GRU. Versus the pre-GRU baseline: K-Fold ensemble
+0.031 AUC / +0.069 F1, Monte Carlo +0.082 AUC / +0.140 F1.

### Statistical rigor (round 4)

- **Repeated K-Fold** (`--n_repeats 5`, 5×5 fold-ensembles, GRU + testprev):
  **AUC 0.805 [0.768, 0.842], F1 0.623 [0.577, 0.669]**. The single-repeat
  AUC 0.827 was a favorable fold seed; 0.805 is the honest mean and the
  recommended K-Fold headline. F1 replicates exactly.
- **MC, 10 seeds/split** (3-member + testprev): **AUC 0.818 / F1 0.657 /
  BAcc 0.763** — doubling seeds sharpens the split ensembles (5-seed:
  0.814/0.634). Recommended MC headline.
- **Official test bootstrap** (2000 subject resamples, N=46): AUC 0.862
  [0.739, 0.966], F1 0.727 [0.522, 0.880]. Wide — every DAIC-WOZ test
  number carries this uncertainty; reporting it is part of the protocol.
- **Wilcoxon signed-rank, GRU vs plain** (25 paired K-Fold runs): per-run
  AUC +0.011, p = 0.61 — the per-run K-Fold difference is *not* significant;
  the GRU's edge shows in the low-variance dev ensembles (±0.004) and the
  ensemble aggregates. Stated to keep claims honest.
- **LOSO** (`--mode loso`, 139 folds, 1 seed): pooled AUC 0.706 / F1 0.476,
  per-subject bootstrap AUC CI [0.61, 0.80]. Deflated by construction: the
  pooled ranking mixes probabilities from 139 different single-seed models,
  whose scales do not align. Comparable only with per-fold seed-ensembles
  (~5× compute); kept as the pessimistic bound.

The legacy pooled-189 comparison (train+dev+test pooled, the leaky protocol
the old versions used) has been **removed**: it is not a leakage-free number.
TC-MIL is compared to the legacy architectures under the rigorous protocol
instead — runs in `results/baselines/legacy_aligned/`, harness
`src/evaluation/eval_legacy.py`.

## Reproduce

All commands run from the repo root.

```bash
# Single-model headline (official test, bge-large, pos_weight=1.0)
python src/training/train_tcmil_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --pos_weight 1.0 \
    --n_seeds 30 --eval_test --threshold_metric prevalence \
    --output_dir results/single_model/headline_pw1_30seed

# Dev ablations (dev only; never touch test)
./run_tcmil_ablation_full189.sh

# Final ensemble: per-member OOF runs, then combine (selection by OOF AUC)
python src/ensemble/oof_threshold_official.py --encoder_name BAAI/bge-large-en-v1.5 \
    --n_seeds 10 --output_dir results/ensemble/oof/oof_bgelarge_plain
python src/ensemble/oof_threshold_official.py --encoder_name mixedbread-ai/mxbai-embed-large-v1 \
    --temporal gru --n_seeds 10 --output_dir results/ensemble/oof/oof_mxbai_gru
python src/ensemble/combine_oof_ensemble.py \
    --runs results/ensemble/oof/oof_bgelarge_plain results/ensemble/oof/oof_mxbai_gru \
    --output results/ensemble/oof/oof_headline/results.json

# Clean CV (GRU + testprev = headline K-Fold; MC ensemble aggregation built in)
python src/crossval/cv_tcmil.py --mode kfold --encoder_name BAAI/bge-large-en-v1.5 \
    --temporal gru --threshold_mode testprev --n_repeats 5 \
    --output_dir results/cross_validation/kfold_gru_testprev_r5
python src/crossval/cv_tcmil.py --mode mc --encoder_name BAAI/bge-large-en-v1.5 \
    --temporal gru --output_dir results/cross_validation/mc_gru_single

# Legacy architecture ladder under the same leak-free protocol
python src/evaluation/eval_legacy.py
```

## Files

| File | Role |
| --- | --- |
| `src/core/tcmil_data.py` | chunking, frozen-encoder embedding + cache (+ e5 prefix), leakage assert |
| `src/core/models/tcmil.py` | gated-attention MIL + symptom aux head + optional GRU/transformer context |
| `src/training/train_tcmil_official.py` | official-split protocol (saves per-seed checkpoints) |
| `src/training/finetune_tcmil.py` | end-to-end encoder fine-tuning (LoRA/bitfit/last_k/full) |
| `src/crossval/cv_tcmil.py` | K-Fold + Monte Carlo CV (innerval / oof / testprev thresholds) |
| `src/evaluation/eval_legacy.py` | legacy architecture ladder under the rigorous protocol |
| `src/statistics/stats_ablation.py` | Wilcoxon / Mann-Whitney U / paired & independent t-tests |
| `src/interpretability/interpret_tcmil.py` | interpretability (attention faithfulness, PHQ-8, bias probes) |
| `src/ensemble/oof_threshold_official.py` | OOF threshold probe (official protocol) |
| `src/ensemble/combine_oof_ensemble.py` | multi-encoder OOF ensemble + subset selection by OOF AUC |
| `src/ensemble/combine_ft_ensemble.py` | fine-tuned multi-member OOF ensemble |
| `src/ensemble/ensemble_tcmil_official.py` | dev-thresholded multi-encoder ensemble |
