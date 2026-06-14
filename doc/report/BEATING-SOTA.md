# Can we honestly match/beat Multi-MTRB & MDSD-FGPL under our protocol?

Assessment (2026-06-13). Our current single model (full 189, leakage-free,
30-seed per-seed): **AUC 0.864, macro-F1 0.774, micro-F1 0.814, pos-F1 0.678.**
Targets: **Multi-MTRB pos-F1 0.88** (2-model MT5+RoBERTa fusion, MIL),
**MDSD-FGPL macro-F1 0.874 / pos-F1 0.828** (2-branch MT5+BERT fusion +
prompt-learning). Both are *fusions* in the prompt-inflated band, neither shows
a participant-only bias control.

## Verdict

- **Multi-MTRB — unverifiable and NOT reproducible; do not treat 0.88 as a
  firm bar.** Three blockers (PDF-verified, see `BASELINE-VERIFICATION.md`):
  (1) the fused 0.88 model reports **no AUC** (the 0.78 is a weaker branch, not
  the fusion — so no ranking win to claim either way); (2) the per-bag instance
  count *n* and threshold *β* are **never given numerically**; (3) the **α
  decision rule is self-contradictory** (majority-positive vs single-negative
  vs any-one-positive across the text and Fig 4). So we **cannot re-run it
  under our protocol** — its 0.88 can be neither confirmed nor refuted. The
  honest stance: flag Multi-MTRB as an unverifiable, under-specified baseline;
  do not chase its number. (Our *fusion = ensemble*, below, remains the fair
  comparator *if* we report against it at all.)

- **MDSD-FGPL (macro 0.874) — the real target; plausible but not guaranteed.**
  It is genuine (acc 0.891, R 0.857, P 0.80; no AUC). The gap is real: our
  macro 0.774 → 0.874 = **+0.10**, and our **oracle threshold ceiling is only
  0.799** — so no threshold/calibration trick reaches it. We must improve the
  **probabilities themselves** (AUC ~0.86 → ~0.90+). Confidence: **plausible**.

## Ingredients (by expected impact)

1. **Encoder fine-tuning — the one big lever we have NOT pulled.** We use a
   *frozen* mean-pooled bge-large; that caps AUC ~0.86. MDSD and Multi-MTRB
   both **fine-tune** their encoders through the task — that is *why* they score
   higher. Our FT suite (`README_FINETUNE.md`, LoRA/last-k with anti-forgetting
   design) is **written but unrun**. This is the credible route to AUC 0.90+
   and macro toward 0.82–0.87.
2. **Two-branch as one system.** Their "single model" *is* a 2-branch fusion
   (MT5+BERT / MT5+RoBERTa). Our 2-encoder system (bge+mxbai, parked) is the
   fair analogue — already macro 0.792; fine-tuned + fused it is the
   apples-to-apples competitor.
3. **Symptom head as a real prediction path** (currently a pure regulariser;
   per-item AUC 0.74–0.94; symptom-sum alone ranks AUC 0.852). Multi-task
   symptom→diagnosis adds independent signal.
4. **Richer instances** — per-symptom / prompt-conditioned chunks (MDSD's
   "fine-grained prompt learning") instead of fixed windows, to sharpen
   separation.
5. **Richer FROZEN encoder** — our encoder ablations only covered ~0.3–0.6B
   models (bge/mxbai/UAE/gte/e5/qwen3-0.6B). A 1.5B-class MTEB-top frozen
   embedder is untried and cheap to test (frozen, cached). Lowest-effort shot
   at raising the AUC ceiling without fine-tuning. **Testing now.**

6. **Ensemble — the fair, last-resort lever against these two specifically.**
   We parked ensembling because it was *unfair vs the single-model baselines*
   (Milintsevich, HCAN). But **MDSD-FGPL and Multi-MTRB are themselves
   2-branch *fusions*** (MT5+BERT / MT5+RoBERTa) — so our ensemble is the
   apples-to-apples competitor *for these two only*. Current 2-member
   (bge-large + mxbai-GRU) already reaches **macro-F1 0.792 / micro 0.830**
   (test, 6 indep ensembles, p=0.006 vs Milintsevich) — closer to MDSD's 0.874
   than the single model, and a *fine-tuned* 2-branch ensemble (lever 1+6)
   is the most likely configuration to actually reach the fusion baselines'
   range under our protocol. Use it *labelled as a fusion*, compared only to
   their fusions — keep the single model as the headline vs single baselines.

## Honest caveats

- 189 subjects is tiny; fine-tuning a 300M+ encoder risks catastrophic
  overfitting (why the FT suite is LoRA + frozen-control + pre-registered
  rules). No guarantee it clears 0.874.
- **The baselines may be protocol-favourable.** The Multi-RoBERTa *branch*
  shows F1 0.83 at AUC 0.78 — a high F1 on modest ranking, hinting the fused
  0.88 leans on a favourable threshold; but the fused model's AUC is
  unreported, so this is a hint, not proof. Reproducing both under our
  protocol is the decisive experiment.

## Bottom line

**Multi-MTRB is unverifiable and irreproducible** (no fused AUC; *n*, β
unreported; α rule self-contradictory) — we cannot confirm, refute, or
reproduce its 0.88, so we **flag it, not chase it**. That leaves **MDSD-FGPL
(macro 0.874)** as the only verifiable fusion target. Reaching its
neighbourhood (~0.82–0.86 macro) is **realistic with encoder fine-tuning**
(+ our 2-branch ensemble / multi-task symptom); beating 0.874 *decisively*
depends on whether it survives leakage-free a-priori thresholding (also worth
checking MDSD for the same reproducibility gaps). Priority: (5) richer frozen
encoder now → (1) fine-tuning → (6) fine-tuned 2-branch ensemble as the
fusion-vs-fusion play. Headline stays the **single model vs the reproducible,
single-model, validity-checked baseline (Milintsevich), which we already beat
significantly.**
