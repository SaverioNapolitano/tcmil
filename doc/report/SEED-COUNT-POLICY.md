# Seed-count policy

Each evaluation stage uses a **different number of seeds on purpose**. The rule:
spend seeds where they reduce the variance of a *reported headline or a
significance test*; do not spend them on a selection step or where another
randomness source already supplies the replication.

| Stage | Seeds | Why this count |
| --- | --- | --- |
| Official / headline test | **30** | One fixed split → seeds are the *only* randomness |
| Monte Carlo (MC) CV | **10** (× 5 splits = 50 runs) | Split resampling is the replication; seeds are secondary |
| OOF threshold probe | **10** | Threshold is a nuisance parameter; already converged |
| Dev design ablations | **5** | Used only to *rank* configs for selection |

## Official / headline test — 30 seeds

The AVEC2017 official split is fixed (mandated test set), so seed
initialisation is the **only** source of randomness. A 30-seed sample is needed
for an honest mean ± std and for valid per-seed significance tests. This count
was raised from 10 after the original single per-seed 0.774 ± 0.029 turned out
to be a **lucky 10-seed group** (seeds 42–51); 30 fresh seeds (100–129) gave the
true 0.751 ± 0.032, and several 10-seed apparent lever gains did **not
replicate** at 30. Anything that becomes a paper headline or enters a t-test
runs at 30.

## Monte Carlo CV — 10 seeds, not 30 (and that is fair)

MC is `StratifiedShuffleSplit`, **5 splits × 10 seeds = 50 model runs**
(`cv_tcmil.py --mode mc --n_splits 5 --n_seeds 10`). The dominant variance
source in MC is **which subjects land in each random 80/20 test split**, *not*
seed initialisation. The 5 resampled splits already supply the statistical
replication — that is the entire point of Monte-Carlo cross-validation. Extra
seeds per split give diminishing returns because inter-split variance dominates.

**Is the seed count vs the 30-seed official comparison fair?** Yes. The two
protocols are never pooled or plotted on the same axis; statistics are computed
*within* each protocol (MC = paired across matched runs; official = independent
30-seed). The seed count is matched to each protocol's variance structure, not
chosen for cross-protocol parity. The comparison that *must* be apples-to-apples
— TC-MIL vs each legacy baseline — runs under the **same** 5 × 10 MC protocol
(same splits, same seeds), so it is fair by construction.

**Should we raise MC to 30 and re-run?** No.
- MC already has 50 runs (> the official 30); 30 seeds would be 150 runs.
- Seeds only tighten an estimate when seed is the variance source — in MC it is
  not, so 3× compute buys no change in conclusions.
- MC significance is already achieved almost everywhere (p < 0.05).
- It would also force re-running every legacy baseline under 30-seed MC to keep
  the head-to-head fair — even more compute for the same answer.

## OOF threshold probe — 10 seeds, not worth 30

The OOF prevalence threshold is a **nuisance parameter** (the decision
threshold), estimated on the 139-subject out-of-fold pool and averaged over
folds × seeds. It already converges and is stable (t ≈ 0.52; no longer collapses
like the 33-subject dev's degenerate t = 0.15). Its variance is ≈ 0.006 and its
total effect on the metric is small (≈ +0.004). Tripling the seeds would not move
the threshold or the downstream test number. The threshold *method* is reported
in the paper; its robustness at 10 seeds is the point.

## Dev design ablations — 5 seeds, not worth 30

Their only job is to **rank configs** for top-2 selection, not to produce a
reportable number. Dev numbers are explicitly **not comparable to test**, and
the final config is frozen by the pre-registered OOF/official protocol, **not**
by dev-argmax. 5 seeds rank the ~18–20 configs reliably enough to pick the
finalists; 30 seeds would be 6× compute (× every config) for a ranking that does
not determine the final result.

## Summary rule

Seeds buy statistical confidence in a *reported* quantity. Headlines and
significance tests on a fixed split need it (→ 30); a protocol whose replication
comes from resampling (MC), a nuisance-parameter estimate (OOF), and a selection
ranking (dev) do not (→ 10 / 10 / 5). Do not "make everything 30."
