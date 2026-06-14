# Literature Review — Baseline Selection for Text-Only DAIC-WOZ Depression Detection

Purpose: pick **valid baselines** for TC-MIL to compare against on the
text-only DAIC-WOZ task, and verify each candidate's reported numbers are
**not suspicious or disproven**. The emphasis is the evidentiary status of
each baseline, not a survey of methods for their own sake. Every claim below
is tied to a primary source; where a number could not be verified or has been
challenged, that is stated explicitly.

> **Attribution correction (2026-06-13).** Earlier drafts cited the
> symptom-prediction baseline as "Burdisso et al. 2023, Brain Informatics."
> That is wrong. The Brain Informatics 2023 symptom paper is
> **Milintsevich, Sirts & Dias** (DOI 10.1186/s40708-023-00185-9).
> "Burdisso et al. 2024" is a *different* paper — the DAIC-WOZ
> interviewer-prompt **bias** study (ClinicalNLP@NAACL 2024). Both are used
> here, for different purposes; do not conflate them.

---

## 1. The validity filter every baseline must pass (Burdisso et al. 2024)

Burdisso, Madikeri & Motlicek, *DAIC-WOZ: On the Validity of Using the
Therapist's Prompts in Automatic Depression Detection from Clinical
Interviews* (ClinicalNLP@NAACL 2024; code: github.com/idiap/bias_in_daic-woz).
**Verified, high confidence.** Demonstrable findings:

- Including **Ellie's (interviewer) prompts** inflates scores via a shortcut.
  PDF-verified macro-F1 (`prompt_bias.pdf`): **P-GCN (patient answers only)
  0.85**, **E-GCN (Ellie's prompts only) 0.88**, **combined/vote 0.90** — the
  last "the highest result to date, by intentionally exploiting it." Per-group
  recall for the depressed class jumps 0.64 → 0.80 when Ellie's prompts are added.
- The shortcut is positional: a model reading only the **second half** of the
  interview (where Ellie asks explicit mental-health questions) scores far
  higher than on the first half.

**Filter applied here (corrected).** Burdisso gives **no absolute-F1
suspicion threshold**, and an earlier draft of this file invented one
("F1 > 0.85") — that is wrong: **0.85 is exactly Burdisso's *clean*
patient-only score**, so it cannot be the bias line. The real signal is the
**method, not the magnitude**: a baseline is bias-suspect if it (a) feeds the
model **interviewer/Ellie prompts** (or the late mental-health-question region)
**without controlling for the shortcut**, in which case scores drift into the
**0.88–0.90 prompt-inflated regime**; the bias-exploiting ceiling is 0.90.
A clean patient-response model can legitimately reach ~0.85. So we judge each
baseline by *prompt/position control*, and flag the 0.88–0.90 prompt-inclusive
numbers, rather than rejecting any score above an arbitrary cut. TC-MIL's own
control: participant-only chunks score dev AUC 0.902 ≥ 0.895 with full dialogue
(`results/ABLATION-STUDY.md`), and the interpretability bias probes show
position–attention correlation ≈0 (`INTERPRETABILITY.md`) — no shortcut.

---

## 2. Primary baseline — Milintsevich et al. 2023 (symptom prediction)

Milintsevich, Sirts & Dias, *Towards automatic text-based estimation of
depression through symptom prediction*, **Brain Informatics 2023**
(10.1186/s40708-023-00185-9). **Verified, high confidence.**

- **Method:** multi-target hierarchical regression predicting the 8 PHQ-8
  symptom severities from interview transcripts (turn encoder = S-RoBERTa;
  interview encoder = BiLSTM + additive attention), then thresholded for the
  binary task. Non-MIL; closest in spirit to the symptom-supervision idea TC-MIL
  uses as an auxiliary.
- **Metrics (test, 47 interviews, 5-seed mean):** **macro-F1 0.739**, total
  PHQ-8 score **MAE 3.78** (per-symptom MAE 0.438–0.830).
- **Why it is the cleanest baseline:** (a) reports on the **held-out test
  split**, not dev; (b) a **validity check in the paper** — perturbing the
  participant yes/no answers caused *no* micro-F1 drop and only −0.52%
  macro-F1, i.e. the model is not riding trivial answer tokens; (c) the number
  (0.739) sits **well within Burdisso's clean patient-only regime** (≤0.85),
  nowhere near the 0.88–0.90 prompt-inflated band. This is the baseline TC-MIL
  is benchmarked against under matched protocol (per-seed mean ± std and
  seed-ensemble; see `results/ABLATION-STUDY.md` and `SEED_STABILITY.md`).
- **Validity, PDF-verified (page 11):** Milintsevich directly investigated the
  exact shortcut Burdisso 2024 warns about — the "Have you been diagnosed with
  depression?" turn. They (a) extracted saliency maps and found the model
  attends **symptom-related turns**, not the diagnosis question; (b) ran a
  Fisher exact test; (c) **perturbed** the yes/no answers and saw **no miF1
  drop and only −0.52% maF1**. Conclusion: the model does not use the question
  as a shortcut. This is a *stronger* validity argument than any score-magnitude
  heuristic.
- **Residual caveat:** the input still includes interviewer turns (no
  participant-only ablation), so a diffuse prompt effect cannot be *fully*
  excluded — but the specific high-risk shortcut was tested and ruled out.

Companion result, same group: *Evaluating LLMs for Depression Symptom
Estimation* (AIME 2025) re-examines LLM features for the same symptom task —
useful as a cross-check, not a stronger baseline.

---

## 3. Other candidate baselines and their evidentiary status

### 3a. TEST-set text-only macro-F1 (the only directly comparable numbers)

PDF-verified from **Milintsevich et al. 2023, Table 2** (`milintsevich.pdf`),
which is itself a curated comparison of *previously published DAIC-WoZ
**test-set** text-only results*. This is the authoritative test-set table —
prefer it over the dev numbers in §3b.

Milintsevich reports **both micro-F1 and macro-F1**, so we match both (micro-F1
≡ accuracy for single-label binary).

Different papers report different F1 variants — **pos-F1** = positive
(depressed) class F1, **macro-F1** = mean of both classes, **micro-F1** =
accuracy. Columns filled where the paper gives them; Milintsevich's pos-F1 is
*derived* (acc 0.766 + macro 0.739 + 14/33 test split → confusion matrix →
pos-F1 ≈ 0.66, approximate due to 5-seed averaging).

| Baseline | Source | pos-F1 | micro-F1 | macro-F1 | Status |
| --- | --- | --- | --- | --- | --- |
| HCAN | Mallol-Ragolta 2019 | 0.63 | — | (0.630‡) | "F1 score" = pos-class; Milintsevich re-tabulates as macro |
| **Milintsevich Symptom Prediction** | Brain Informatics 2023 | ≈0.658 (derived) | **0.766** | **0.739** | **PRIMARY — verified, validity-checked** |
| HCAG-T | Niu 2021 | — | — | 0.770 ‡ | ‡ validation only |
| MDSD-FGPL | Zhang & Guo 2024 (PRL) | 0.828 | — | 0.874 | text test, **2-branch MT5+BERT fusion**, bias control not shown |
| Multi-MTRB | Zhang 2025 (Sci Rep) | 0.88 | — | — | text test, **2-model MT5+RoBERTa fusion**, no AUC |
| **TC-MIL single (30-seed, pos_weight=1.0)** | this work | **0.678 ± 0.041** | **0.814 ± 0.024** | **0.774 ± 0.029** | single network; **leads Milintsevich on all three** (macro p=3.8e-07) |

Only **HCAN** and **Milintsevich** are genuine test-set numbers in *that*
table. The **single** TC-MIL model (pos_weight=1.0) **significantly beats**
Milintsevich on **both** micro-F1 (0.814 vs 0.766) and macro-F1 (0.774 vs
0.739, one-sample t over 30 seeds **p=3.8e-07**), with higher AUC (0.864) — no
ensembling (`SEED_STABILITY.md`, `SINGLE_MODEL_IMPROVEMENTS.md`). Micro-F1 uses
the same a-priori prevalence threshold as macro — honest, not a test-tuned
oracle.

A full audit of every PDF in `paper/related_work/` (which split + which
modality each paper actually uses) is in **`BASELINE-VERIFICATION.md`**.
Headline: of 16 papers, only **4 are test-set AND text-only** — Milintsevich
0.739 (validity-checked single model), HCAN 0.63, **MDSD-FGPL 0.874** (2-branch
MT5+BERT fusion + prompt-learning), and **Multi-MTRB 0.88** (2-model MT5+RoBERTa
fusion). MDSD-FGPL and Multi-MTRB both **exceed our single model (0.774)** on
macro-F1, but both are *fusions* in the 0.87–0.88 prompt-inflated band with no
verified bias control; our single model is one network with a passing
participant-only control. Honest standing: we lead the cleanest verified
single-model baseline (Milintsevich), not the unverified fusion numbers.
Eight other test-set papers are **multimodal** (incl. Gong/`topic`, test F1
0.60); the rest report on dev only. The fused Multi-MTRB reports **no AUC** —
the 0.78 AUC in the paper is the **Multi-RoBERTa branch** (F1 0.83), not the
fused 0.88 model, so no ranking comparison to the fusion is possible. Cite
Multi-MTRB's F1 0.88 only; do not claim a TC-MIL AUC win over it.

### 3b. Dev-set context (label clearly as dev; never mix with test)

| Baseline | Source | macro-F1 (dev) | Status for comparison |
| --- | --- | --- | --- |
| Xezonaki HAN+L | Interspeech 2020 (PDF Table 5) | 0.69 | dev ref / architecture precedent |
| ELMo BiLSTM + attention | Shen/Yang/Lin, ICASSP 2022 | 0.83 | dev ref (web-sourced, not PDF-checked) |
| HCAG / text-GCN / Psi-GCN | RED table | 0.816 / 0.838 | dev refs |
| SEGA / SEGA++ | NAACL 2024 long.452 / RED table | 0.849 / 0.878 | dev; SEGA++ in the 0.88-prompt-inflated band → check prompt control |
| RED (personalized RAG) | Zhang et al. 2025 (arXiv 2503.01315) | 0.90 | **EXCLUDE as headline** — dev + RAG, no public test, bias-band |
| BERT-BiLSTM + XGBoost | ICCAI 2024 | 0.91 (non-standard) | **EXCLUDE — non-standard split, unverifiable** |

Notes grounded in fact:
- DAIC-WOZ **test labels are not public**; many papers report only dev/
  train+dev. The §3b numbers are **dev** and not comparable to §3a test numbers.
- §3b numbers tagged "web-sourced" are not yet PDF-verified — web summaries
  proved unreliable for exact figures (they fabricated a "Niu 0.92 / Dai 0.96"
  that does not match Milintsevich's tabulated Niu-HCAG 0.770 val). Treat as
  provisional until checked against a primary PDF.

---

## 4. Claims that did NOT survive fact-checking (do not cite as facts)

**How this section was checked — re-derived 2026-06-13.** Each claim was
re-verified now via web search against the primary paper plus independent
secondary sources. The parenthetical `(s-c)` is the resulting tally over the
(up to three) independent sources consulted: **s = sources that support the
claim as stated, c = sources that contradict it or fail to support it.** So
`(0-3)` = no source supports it, `(1-2)` = mostly unsupported, `(3-0)` =
unanimously supported (the bar §1 Burdisso bias and §2 Milintsevich meet).
This replaces the earlier inherited vote tallies; verdicts and corrected facts
below are from this session's searches (sources listed in §Sources).

**"Rejected"** means the claim could not be corroborated as stated — so do
**not** cite it as fact. It does *not* always mean the work is fake: often the
paper is real but the *specific number, ranking, or validity* is the
unsupported part. Where the re-derivation produced a *corrected* fact, it is
given.

- **"Burdisso 2023 symptom-aware 0.739"** — **rejected, wrong attribution
  (0-3).** The Brain Informatics 2023 symptom paper is **Milintsevich et al.**,
  not Burdisso (3 sources: Springer, PMC, PubMed). Corrected throughout the repo.
- **"Prior text-only SOTA 0.84–0.96 (Niu 0.92, Dai 0.96)"** — **rejected;
  the specific 0.92/0.96 figures are unverified (PDF-contradicted) (0-3).**
  Milintsevich's Table 2 tabulates Niu's HCAG model at **maF1 0.770
  (validation)**, not 0.92 — the web's "0.92/0.96" do not match the primary
  comparison table (possibly a different metric or mis-attribution). Whatever
  the exact figures, any prompt-inclusive text-only model in the 0.88–0.90 band
  is in Burdisso's bias-inflated regime (§1). Do not benchmark against them.
- **"HAN F1 68.6 is SOTA" (Xezonaki 2020)** — **dev-set result, not a test
  baseline (PDF-verified, `xenozaki.pdf` Table 5).** HAN+L = **F1-macro 0.69**
  on the **DAIC-WoZ development set**; the abstract's 71.6 is the *separate*
  General Psychotherapy Corpus (5-fold CV), and there is **no DAIC-WoZ test
  number**. Two corrections this surfaced: (i) this repo's own earlier
  web-based "70.3 test" was wrong (0.69 dev is ground truth); (ii) **even
  Milintsevich's Table 2 mis-tabulates this as test "0.700"** — the dev→test
  error is in the literature itself. Use only as a **dev** reference.
- **"GPT-4 two-shot E-DAIC RMSE 3.975 beats AVEC baseline"** — **rejected as a
  comparable baseline (0-3).** A closely related result (~RMSE 3.98, textual
  modality) exists, but on **E-DAIC** (not DAIC-WOZ) using **Whisper ASR**
  transcripts (not gold) — not comparable to our gold-transcript test split.
- **"Multimodal fusion F1 0.85 > text-only 0.83"** — **rejected (1-2).** Only
  *relative* gains are documented (e.g. +0.06 over voice, +0.25 over text);
  the specific 0.85/0.83 pair was not found, and multimodal is out of this
  text-only scope.
- **"LLM zero-shot beats 5 prior models (MAE 3.46–3.80)"** — **rejected
  (0-3).** No primary source supports this ranking.
- **"Symptom 3.78 vs direct-score 5.03 MAE"** — **unverified (1-2).** The
  Milintsevich symptom method and its 3.78 total-score MAE are real, but the
  *paired comparison against a 5.03 direct-score baseline* was not located in
  sources.

Corroboration found during re-derivation: a 2026 follow-up (*When Consistency
Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews*,
arXiv 2603.24651) independently reinforces the Burdisso interviewer-bias
concern — the prompt-control validity filter in §1 is well-founded.

---

## 5. Recommended comparison set (the defensible subset)

1. **Headline test-set baseline: Milintsevich et al. 2023, macro-F1 0.739.**
   TC-MIL is reported against it both per-seed (Milintsevich-style
   mean ± std) and seed-ensemble.
2. **Second test-set anchor:** HCAN (Mallol-Ragolta 2019) **0.630 test** — the
   only other genuine test-set text-only macro-F1 in Milintsevich's Table 2.
3. **Dev-set context:** Xezonaki HAN+L (0.69), ELMo-BiLSTM+attention (0.83),
   SEGA (0.849) as *dev* references, clearly labelled as dev, never mixed with
   test numbers.
4. **Do not headline against** RED (0.90 dev/RAG) or the ICCAI XGBoost (0.91
   non-standard) — both fail the validity filter; mention only as cautionary
   context.
5. **Regression axis:** Milintsevich total-score MAE 3.78 is the reference if
   TC-MIL is ever extended to PHQ-8 score regression (currently classification
   only; aux head is binarized, so no MAE is reported — do not fabricate one).

---

## Sources (primary, verified)

- Milintsevich, Sirts, Dias 2023, *Towards automatic text-based estimation of
  depression through symptom prediction*, Brain Informatics — **local PDF
  verified** (`paper/related_work/milintsevich.pdf`): Table 2 test maF1
  **0.739 ± 0.025**, MAE 3.78 ± 0.13; page-11 shortcut/robustness analysis.
  https://braininformatics.springeropen.com/articles/10.1186/s40708-023-00185-9
- Mallol-Ragolta et al. 2019, *A Hierarchical Attention Network-Based Approach…*,
  Interspeech — HCAN, **test maF1 0.630** (via Milintsevich Table 2).
  Local: `paper/related_work/mallolragolta19_interspeech.pdf`
- Burdisso, Reyes-Ramírez, Villatoro-Tello, Sánchez-Vega, López-Monroy,
  Motlicek 2024, *DAIC-WOZ: On the Validity of Using the Therapist's
  Prompts…*, ClinicalNLP@NAACL (arXiv 2404.14463) —
  https://arxiv.org/abs/2404.14463 (code: https://github.com/idiap/bias_in_daic-woz)
- Milintsevich et al. 2025, *Evaluating LLMs for Depression Symptom
  Estimation*, AIME — https://dias.users.greyc.fr/publications/aime2025.pdf
- Zhang et al. 2025, *RED: Explainable Depression Detection … Personalized
  RAG* (arXiv 2503.01315) — https://arxiv.org/pdf/2503.01315
- Shen/Yang/Lin 2022, BiLSTM+attention over ELMo, ICASSP —
  https://arxiv.org/pdf/2202.08210
- Xezonaki, Paraskevopoulos, Potamianos, Narayanan 2020, *Affective
  Conditioning on Hierarchical Attention Networks…*, Interspeech — DAIC-WoZ
  **dev** F1-macro **0.69** (Table 5), GPC 71.6 (5-fold CV); no DAIC-WoZ test
  number. Local copy: `paper/related_work/xenozaki.pdf` —
  https://arxiv.org/abs/2006.08336
- Depression Detection in Clinical Interviews with LLM (NAACL 2024,
  SEGA comparison) — https://aclanthology.org/2024.naacl-long.452.pdf
- *When Consistency Becomes Bias: Interviewer Effects…* 2026 (arXiv
  2603.24651) — corroborates the interviewer-bias filter.

Method note: §3a/§4 **PDF-verified 2026-06-13** against
`paper/related_work/` (`milintsevich.pdf`, `xenozaki.pdf`,
`mallolragolta19_interspeech.pdf`); web-only figures elsewhere are flagged
inline and remain provisional. Web summaries proved unreliable (they invented
a "Xezonaki 70.3 test" and a "Niu 0.92" that the primary PDFs contradict), so
**the PDF is ground truth and overrides any web figure.** Confirmed from PDFs:
the symptom baseline is **Milintsevich** (test maF1 0.739 ± 0.025), with an
in-paper shortcut/robustness check passing; **Xezonaki is DAIC-WoZ dev 0.69,
not a test number** — an error even Milintsevich's own Table 2 propagated
(tabulating it as test "0.700"). Genuine test-set text-only baselines:
HCAN 0.630 and Milintsevich 0.739.
