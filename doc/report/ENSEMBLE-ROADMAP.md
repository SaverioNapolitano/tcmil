# Ensemble improvement roadmap (beyond fine-tuning)

Fine-tuning (cluster) is tracked separately; everything here is frozen-encoder.

## STATUS (2026-06-14)

**Executed (roadmap v1):** best-PAIR selection + per-member temperature scaling +
logit/arith rule, leakage-free on dev (`ensemble_pair_search.py`, members in
`results/ensemble/pw1_members/`). Winner = **bge-large GRU + UAE-large GRU**,
arith mean, prevalence threshold → **6×(5+5) macro 0.833 / micro 0.862 / AUC
0.868** (full-30 0.803/0.838/0.864). This is the current best (was 0.809 for the
bge-plain+mxbai-GRU pair). Both members GRU, 2 distinct encoders → fair.

**Data available offline:** each member saves `dev_prob_runs` + `test_prob_runs`
(+ labels). NO OOF probs, NO model checkpoints saved. → OOF-threshold and
MC-dropout levers need NEW compute; α-weight and Platt are pure-offline.

## Complete lever map (original L1–L5 → A/B/C, nothing dropped)

| original lever | status | bucket |
| --- | --- | --- |
| L1 best-pair selection | DONE — bge-gru+uae-gru, test 6×5+5 macro **0.833** | — |
| L2 per-member calibration (temp) | DONE | — |
| L3 combination rule (logit/arith) | DONE — arith | — |
| L4 OOF threshold | DEFERRED (needs per-member OOF runs; low EV) | A2 |
| L5a multi-window view | DONE → REJECTED (B4 below) | B4 |
| L5b participant-only-view member | **NOT YET RUN** | B5 |
| L5c MC-dropout TTA | was BLOCKED (no ckpts); now unblockable (train saves ckpts) | C |

So C contains only L5c. Don't forget **L5b** (ponly-view member).

### B4 result (L5a) — REJECTED (2026-06-14, `eval_b4.py`, `results/ensemble/eval_b4.txt`)
bge-gru@w4 + uae-gru@**w6**: member prob-corr 0.960→**0.902** (the w6 view DID
decorrelate) BUT uae-gru@w6 is individually weaker + high-variance → 6×5+5 macro
**0.783** (−0.049 vs 0.833), full-30 0.771±0.077. AUC 0.873 (≈). Decorrelation
gain < member-quality loss. Baseline (both @w4) stays best. Note the w4 pair is
0.96-correlated → little ensemble headroom; that's the core difficulty.

### B5 result (L5b ponly-view member) — REJECTED (2026-06-14, `eval_b4.py`)
bge-gru@w4(dialogue) + uae-gru(participant-only): member corr 0.960→**0.826**
(most decorrelated yet) BUT 6×5+5 macro **0.694** (−0.139), full-30 0.710±0.048.
The ponly member is individually far weaker (no interviewer context) → ensemble
craters. Same failure mode as B4, worse.

**PATTERN (B4+B5):** forced-diversity via weaker views/configs decorrelates but
the member-quality loss dominates → ensemble drops. The only diversity lever that
decorrelates *without* sacrificing member accuracy = **NCL (B6, joint training)**
— the decisive test.

### B6 result (NCL joint training) — REJECTED (2026-06-14, `train_ncl_pair.py`)
λ=0.5, 30-seed: member corr 0.960→**0.901** (decorrelated) but ensemble 6×(5+5)
macro **0.823±0.011** / full-30 0.814±0.022 — **< independent pair 0.833**. The
1-seed smoke (0.847) was a lucky seed; the 30-seed truth is no gain. NCL joins
B4/B5 as decorrelation-without-net-benefit.

## CONCLUSION (diversity program, 2026-06-14)
A (re-weight/Platt), B4 (multi-window), B5 (ponly-view), B6 (NCL) **all ≤ 0.833**.
The independent equal-weight, temperature-scaled **bge-gru + uae-gru @ 6×(5+5)
macro 0.833** remains the FINAL best ensemble. Root cause: the two allowed
encoders are similar (both ~1024-d bidirectional + GRU, base corr 0.96), so
forced diversity always costs more member quality than it recovers, and the
independent average is already near the achievable ceiling. Remaining real lever
= fine-tuning (cluster). MC-dropout (C, now unblocked) is the only cheap untried
item; expect ≤ noise.

## Execution plan (ordered, per 2026-06-14 decision)

Order: **C-leftovers (roadmap, not in A/B) → A → B**.

### C — roadmap leftover, not in A/B
- **MC-dropout / TTA per member** — BLOCKED: members save probs only, no
  checkpoints. Needs retraining all members with `torch.save` + stochastic
  inference. Deferred (compute); not run before A.

### A — refine the combination — DONE, REJECTED (2026-06-14, `ensemble_refine.py`, `results/ensemble/refine_A.txt`)
1. **Weighted α** — REJECTED. For the diverse pair bge-gru+uae-gru (temp, arith),
   α=0.5 is best on BOTH dev (0.821) and test (0.833); any α≠0.5 only hurts
   (α=0.3→.823, 0.6→.823, 0.7→.818). Full search over pairs×calib×rule×α
   *overfit* the n=35 dev → degenerate winner bge-gru+mxbai-plain α=1.0 (= single
   member, dev 0.827) generalised WORSE (test 6×5+5 0.823 < baseline 0.833).
2. **OOF threshold** — DEFERRED (needs per-member 139-fold OOF runs; single-model
   OOF lever was only +0.004 NS → low EV).
3. **Platt per-member** — REJECTED (did not win dev selection; same overfit risk).
   **Conclusion: equal-weight (α=0.5) temperature-scaled bge-gru+uae-gru @ 0.833
   stays best.** Members too similar to re-weight; gains must come from B.

### B — increase 2-branch diversity (after A; biggest EV, current pair too alike)
4. **Heterogeneous bag-views**: bge-gru@w4 + uae-gru@w6 (retrain uae-gru@w6).
5. **Different head/objective per branch** (GRU+plain, or aux-heavy + aux-light).
6. **Negative-correlation learning** — joint-train members with a decorrelation
   penalty. Biggest potential, most code.

> Caveat: n=47 test → wide CI; at 0.833 vs single-oracle ~0.80 ceiling, A is
> likely within seed noise. B is the higher-EV bet (decorrelate the two
> branches). Promote a change only if it clears 0.833 beyond seed noise.

---

## Original lever inventory (reference)

Levers to push the pos_weight=1.0 ensemble past the previous
**macro-F1 0.809 ± 0.019 (6×5+5)** (2-member: bge-large plain + mxbai GRU).

## Hard constraint — 2 encoders only in the FINAL model

The ensemble is benchmarked against **2-branch fusion systems** (MDSD-FGPL,
Multi-MTRB). Using >2 encoders would make that comparison unfair. So:

- **Search freely** over combinations (any encoder × {plain, GRU}) to find the
  best-decorrelated pair.
- **Final reported ensemble = exactly the two best combined encoders**, no more.
- "Two members" may still each be a seed-ensemble of the same encoder — that is
  one branch, fair. The constraint is on the number of *distinct encoders /
  branches* (≤2), matching the fusion baselines.

## Levers (priority order)

### 1. Best-pair member selection (biggest safe lever)
Ensemble gain ∝ member **de-correlation**, not individual accuracy. Pick the
pair that maximises OOF AUC / minimises prob-correlation, not top-1 each.
- Candidate encoders (full189 dev, see design-ablation table): bge-large,
  mxbai-large, UAE-large (dev 0.910, best non-bge), all 1024-dim.
- Architecture axis adds free diversity: **plain** attention is near-uniform /
  linear, **GRU** attention is selective (Fig. 3) → different inductive bias →
  decorrelated errors even with the same encoder.
- Method: rerun the candidate encoders as pw1.0 30-seed members WITH
  `--eval_test` (so test_prob_runs saved), then **OOF forward-selection of the
  best PAIR** (the original 0.807 OOF-AUC pick mechanism), constrained to 2
  encoders. Report only the winning pair.
- Likely top candidate to add to the search: **UAE-large + GRU**.

### 2. Per-member calibration before averaging
pw1.0 is still BCE-recall-skewed → member probs are mis-scaled, which hurts the
mean. **Temperature-scale each member on OOF**, then average. Cheap,
leakage-free. Matters more for the ensemble than for the single model (scale
mismatch between two members directly distorts prob-averaging).

### 3. Combination rule beyond plain mean
- **Logit / log-odds (geometric) mean** instead of arithmetic prob-mean — often
  better when members are confident-but-imperfect.
- **OOF-AUC-weighted** average (light, not learned).
- Keep any meta-model SIMPLE: n=139 OOF / n=47 test → stacking / logistic
  meta-learner **overfits fast**. If stacking at all: CV-only, 1–2 params max.

### 4. OOF threshold for the ensemble
Currently a-priori prevalence (0.28). Ensemble probs are tighter → an
**OOF-tuned threshold** (matched train sizes, leakage-free) historically gave
+F1. Low risk.

### 5. Cheap input diversity (free members, no new encoder)
Useful in the SEARCH; final still collapses to ≤2 encoders.
- **multi-window** members (w2 / w4 / w6) — window-sensitive per ablation.
- **participant-only vs dialogue** view (ponly dev AUC 0.902 ≥ dialogue 0.895).
- **MC-dropout test-time** averaging (MC already AUC 0.883) → cheap extra
  pseudo-members per model.

## Skip / low-value
- More seeds (variance already ~0.006).
- Snapshot / checkpoint ensembling (marginal at n=47).

## Recommended sequence
1. Add **UAE-GRU** pw1.0 30-seed member (`--eval_test`); run OOF best-PAIR
   selection over {bge,mxbai,uae}×{plain,GRU}, output the winning 2-encoder pair.
2. **Per-member temperature scaling** on OOF, re-average.
3. **Logit-mean + OOF threshold** on the chosen pair.
Steps 1–2 are the likely real lift; 3–4 are polish under wide n=47 CIs.

> Caveat: every DAIC-WOZ test number sits on n=47 → wide bootstrap CIs. Only
> promote an ensemble change if it clears the current 0.809 by more than seed
> noise (one-sample t, as in `combine_pw1_ensemble.py`).
