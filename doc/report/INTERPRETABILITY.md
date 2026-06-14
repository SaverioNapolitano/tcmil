# TC-MIL Interpretability Analysis — Methodology

Interpretability here is treated as a *measured* result, not an illustrative
heat-map. Every claim is quantified, validated for faithfulness, reported
with cross-seed stability, and kept leakage-clean (models trained on the
official train split; analysis on dev, or once on test for the final
figures). Engine: `training/interpret_tcmil.py` → `results/interpretability/`.

## Why these analyses

TC-MIL has three interpretable surfaces, and the design of the study uses all
three rather than only attention:

1. **Gated-attention over dialogue chunks** — the native MIL explanation
   (Ilse & Welling 2018): which interview segments drive the bag decision.
2. **PHQ-8 symptom auxiliary head** — a clinical read-out absent from generic
   text classifiers; lets us ask *which depression construct* the text and
   the attention align with.
3. **Dialogue-chunk instances** — human-readable, so a top-attended chunk is
   directly inspectable (unlike utterance-level or token-level bags).

## The analyses (and the literature they answer to)

### 1. Attention, with stability
Raw gated-attention weights per chunk, **averaged over the seed ensemble**,
and a **cross-seed stability** score (mean pairwise Spearman of the per-seed
attention vectors). Rationale: on n=107 a single seed's attention is noisy;
an explanation that reshuffles every seed is not trustworthy. We report the
stability number rather than hiding it.

### 2. Faithfulness, not just saliency
Attention weights are *not* assumed to be explanations (Jain & Wallace 2019).
We validate them against occlusion using the ERASER metrics
(DeYoung et al. 2020):

- **Comprehensiveness@k** = `p(full) − p(remove top-k attended chunks)`.
  High ⇒ the attended chunks are the ones that carry the decision.
- **Sufficiency@k** = `p(full) − p(keep only top-k)`. Near 0 ⇒ the top-k
  chunks alone reproduce the decision.
- **Attention/occlusion correlation** = Spearman between each chunk's
  attention and its leave-one-out Δprob. Positive ⇒ attention ranks chunks
  the way ablation does, i.e. attention is faithful here.

These convert "the model attends to X" into "removing X actually changes the
prediction," which is the bar a clinical explanation must clear.

### 3. Clinical read-out (PHQ-8 symptom head)
- **Per-item AUC** across the split: which of the 8 PHQ-8 symptoms are
  text-predictable at all (some, e.g. appetite/sleep, are more lexicalised
  than others).
- **Per-interview predicted vs ground-truth symptom profile**: a clinician-
  facing artifact, and a check that the aux head learned symptom structure
  rather than a single depression axis.

### 4. Attention ↔ symptom link (specific to this project)
For each interview's most-attended chunk, the symptom head is applied to that
chunk's own contextualized representation; we record which PHQ-8 item it most
activates. Aggregated, this answers "when TC-MIL bases a decision on a chunk,
which clinical construct is that chunk speaking to?" — a bridge between the
attention explanation and the diagnostic criteria that no plain classifier
offers.

### 5. Validity / bias probes (Burdisso et al. 2024)
The known DAIC-WOZ text shortcut lives in (a) interviewer prompts and (b) the
second interview half (where Ellie asks explicit mental-health questions). We
probe whether TC-MIL exploits it:

- **Position bias**: Spearman(attention, normalized chunk position). A strong
  positive value would mean the model leans on the late mental-health-question
  region (the shortcut).
- **Interviewer-content bias**: fraction of attention mass on chunks whose
  *interviewer* line contains a PHQ-8-symptom keyword (lexicon in the engine).
- **Dialogue vs participant-only**: rerun with `--participant_only` and
  contrast attention/faithfulness — complements the accuracy-level bias
  control already in the ablation (participant-only dev AUC 0.902 ≥ dialogue
  0.895). Interpretability shows *where* the robustness comes from.

## Leakage & rigor rules
- Models trained on the **official train split only**; dev used for early
  stopping. Analysis defaults to `--split dev`; `--split test` is for the
  report-once final figures.
- No analysis statistic feeds back into training or threshold selection.
- Faithfulness uses the model's own forward passes (occlusion), not a
  surrogate — it cannot be gamed by attention.
- Seed ensemble (default 5) so attention/stability are not single-seed
  artifacts.

## Run

```bash
# Development analysis (safe to iterate)
python training/interpret_tcmil.py --split dev --n_seeds 5

# Bias control: same, interviewer lines removed
python training/interpret_tcmil.py --split dev --participant_only --n_seeds 5

# Final, report-once figures
python training/interpret_tcmil.py --split test --n_seeds 5
```

Outputs `results/interpretability/interpret_<split>.json`:
`summary` (stability, faithfulness, bias probes, per-symptom AUC,
attention↔symptom histogram) and `per_interview` (probabilities, top-3
attended chunks with text, predicted vs true PHQ-8 profile).

## Results (dev, 5-seed; `results/interpretability/`)

| Metric | Dialogue | Participant-only |
| --- | --- | --- |
| Attention cross-seed stability (Spearman) | 0.77 | 0.66 |
| Attention↔occlusion rank corr (faithfulness) | **−0.23** | 0.00 |
| Comprehensiveness @1 / @5 | 0.09 / 0.28 | 0.06 / 0.29 |
| Position–attention corr (Burdisso shortcut) | **+0.03** | +0.32 |
| Interviewer-keyword attention mass | 0.71 | 0.00 (by construction) |
| Symptom-head AUC (range over 8 items) | 0.74–0.94 | 0.75–0.91 |

Honest reading:

1. **Attention is stable but NOT per-chunk faithful.** Cross-seed stability is
   high (0.77), yet the attention↔occlusion rank correlation is **negative**
   (−0.23) for the dialogue model: the single chunk the model attends to most
   is *not* reliably the one whose removal most changes the prediction. This is
   the Jain & Wallace caveat, confirmed here — **do not present a single-chunk
   attention heat-map as "the explanation."** What *does* hold is aggregate
   **comprehensiveness** (removing the top-k attended chunks lowers the
   probability, 0.09→0.28 as k grows), so attention is informative in bulk, not
   per-instance. Report attention with this caveat, not as a faithful saliency.
2. **No Burdisso position shortcut.** Position–attention correlation is ≈0
   (+0.03) for the dialogue model — attention is spread across the interview,
   not concentrated in the late mental-health-question region. This is the
   strongest validity result: combined with participant-only dev AUC 0.902 ≥
   0.895, TC-MIL does not ride the second-half shortcut. (Participant-only
   shows a mild +0.32 — with interviewer cues removed it leans slightly later,
   but still far from a hard shortcut.)
3. **The symptom head is genuinely clinical.** Per-item AUC 0.74–0.94, highest
   on the most lexicalised symptoms (No-Interest 0.93, Tired 0.94, Depressed
   0.84/0.91). The PHQ-8 read-out is real, not decorative.
4. **Attention concentrates on sleep / fatigue / mood chunks** (attention↔
   symptom link dominated by Sleep+Tired in dialogue, Tired+Depressed in
   participant-only) — clinically sensible targeting.

**Test-split confirmation** (`results/interpretability_test/`, report-once):
attention stability 0.79, occlusion-rho −0.21, comprehensiveness@5 0.32, and
**position–attention correlation 0.002** — the no-shortcut and
stable-but-not-faithful findings replicate on held-out test. (Per-symptom AUC
is unavailable on test: the test split CSV does not include the PHQ-8 *item*
labels, only the binary label.)

**pos_weight = 1.0 re-test** (`results/interpretability_pw1/`): the hypothesis
that de-biasing the loss would make attention *more* faithful is **not
supported** — vs the auto-pos_weight model, attention got *less* stable
(0.60 vs 0.77) and *less* faithful (occlusion-rho −0.32 vs −0.23, comp@5 0.23
vs 0.28). The recall-biased loss apparently produced *peakier* (more
selectable) attention; removing it improves the decision (macro-F1 0.751 →
0.774) but slightly blurs the per-chunk attribution. The **no-shortcut**
result is unchanged (position–attention rho −0.04 ≈ 0). Net: faithfulness and
F1 are decoupled here; report the F1 win on its own terms, keep the attention
caveat.

Caveat on faithfulness sign: sufficiency is slightly negative (a top-k-only
bag scores *above* the full bag) because the recall-biased pos-weighted model
fires on any depressive chunk; this is consistent with the negative occlusion
correlation and is reported, not hidden.

## How to read the headline numbers
- **attn_occlusion_rho_mean > 0** and **comprehensiveness ≫ 0**: attention is
  faithful — safe to show attention figures as explanations.
- **position_attention_rho ≈ 0 (not strongly positive)** and modest
  **interviewer_keyword_attention_mass**: the model is *not* riding the
  Burdisso shortcut — the strongest validity argument for a text-only
  DAIC-WOZ model.
- **symptom_head_auc** high on lexicalised items: the clinical read-out is
  real, not decorative.
