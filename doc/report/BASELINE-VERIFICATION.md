# Baseline Verification — Which `paper/related_work/` PDFs Evaluate on the DAIC-WoZ TEST Set

Every PDF in `paper/related_work/` was inspected (full read for the load-bearing
ones, text-extraction grep for the rest) to record **(a) the evaluation split
actually used for the headline number and (b) the modality.** Only **test-set,
text-only** results are directly comparable to TC-MIL.

## Result (16 papers)

| Paper (file) | Eval split | Modality | Headline | Comparable? |
| --- | --- | --- | --- | --- |
| `milintsevich` | **TEST** | text | macro-F1 0.739 ± 0.025 | ✅ primary text-only test baseline |
| `mallolragolta19` (HCAN) | **TEST** | text | F1 0.63 / UAR 0.66 | ✅ text-only test baseline |
| `multilevel` (MDSD-FGPL) | **TEST** (+val) | text | **macro-F1 0.8741** (binary F1 0.828) | ⚠️ text test, but a **2-branch fusion** (MT5 0.830 + BERT 0.852 → FGPL 0.874) in the prompt-inflated band; prompt-control not shown |
| `mil_bio` (Multi-MTRB, MIL) | **TEST** | text | F1 **0.88** (no headline AUC) | ⚠️ text test, F1 in Burdisso's 0.88-prompt-inflated band (prompt control not shown); see AUC note |
| `rohanian19` | **TEST** | multimodal | F1 0.81 (text-only 0.69) | ✗ multimodal |
| `17_agent_based_splitting` | **TEST** (107/35/47) | **TEXT** (their modality col = "T") | **MV-IA-Mean macro-F1 0.730** (dev 0.69), UAR 0.72, binary BCELoss | ✅ **COMPARABLE — text-only, official test, binary macro-F1.** Was WRONGLY excluded as "multimodal": multi-**view** (patient/therapist split of the transcript) ≠ multi-**modal**. Agarwal, Dias, Dollfus, PAI4MH @ NeurIPS 2022. We beat it (0.774). NOW in tab:external. |
| `2024.clpsych-1.9` | **TEST** | **TEXT** (multi-view discourse) | verify number | ⚠️ also Agarwal/Dias/Dollfus TEXT-only (CLPsych 2024, "Analysing Relevance of Discourse Structure"), DISTINCT from 17_agent (not "same work"). Text-only — re-verify its test macro-F1; candidate baseline. |
| `multimodal2` | **TEST** | multimodal | F1 0.79 (val 0.93 — overfit gap) | ✗ multimodal |
| `rf` (random forest) | **TEST** (+dev) | multimodal | RMSE/MAE (regression) | ✗ multimodal/regression |
| `svm` | **TEST** (+dev) | multimodal | F1 (dev 1.00 / test 0.63) | ✗ multimodal |
| `topic2` (TOAT) | **TEST** (+val) | multimodal (A/V/T) | best-in-table | ✗ multimodal |
| `xenozaki` (HAN+L) | **DEV** | text | macro-F1 0.69 | ✗ dev, not test |
| `Alhanai` | **DEV** | multimodal | F1 0.77 | ✗ dev (test labels withheld), multimodal |
| `prompt_bias` (Burdisso bias) | **DEV** | text/GCN | bias study; "test labels not public" | ✗ dev; the validity-filter paper |
| `topic` (Gong) | **TEST** (also CV + dev; Table 3) | multimodal | F1 0.60 test (RMSE 4.99) | ✗ multimodal (audio+video+LIWC), regression-first |
| `multimodal` | unclear / non-standard | multimodal | F1 **0.97**, MAE 2.62 | ✗ implausibly high → non-standard split, suspect |

## Conclusions

- **Not all papers use the official test set.** 12/16 report on the AVEC2017
  test split; 4 do not: `xenozaki` (dev), `Alhanai` (dev), `prompt_bias`/
  Burdisso (dev — states test labels are not public), and `multimodal`
  (F1 0.97 — non-standard/suspect). [`topic`/Gong **does** report a test
  number — F1 0.60 — alongside its CV and dev columns; corrected 2026-06-13.]
- **Only 4 are test-set AND text-only** — the directly comparable set:
  Milintsevich **0.739** (validity-checked single model), HCAN **0.63**,
  MDSD-FGPL **0.874** (2-branch MT5+BERT *fusion*, prompt-learning, prompt
  control not shown), and Multi-MTRB **0.88** (also a 2-model MT5+RoBERTa
  fusion). **MDSD-FGPL and Multi-MTRB both report higher macro-F1 than our
  single model (0.774)** — but both are *fusions* in the 0.87–0.88
  prompt-inflated band without a verified bias control, whereas our single
  model is one network with a passing participant-only control. The other
  7 test-set papers are **multimodal** (audio/visual) and are not a fair
  text-only comparison.
- **A field-wide caveat:** Burdisso (`prompt_bias`) states the DAIC-WoZ test
  labels were not publicly available (AVEC competition). Other groups clearly
  do report test numbers (the AVEC2017 test labels were released), so there is
  genuine inconsistency in the literature about test-label availability —
  another reason to verify each paper rather than trust a leaderboard.
- **`mil_bio` (Multi-MTRB) — NOT faithfully reproducible; treat 0.88 as
  unverifiable.** Three blockers (PDF-verified):
  1. **No AUC for the fused model** (Table 2 = Acc/F1/P/R). The only AUC
     (0.78, Fig 15) is the **Multi-RoBERTa *branch*** (Table 4/5: Acc 0.85,
     F1 0.83) — not the fused 0.88 model. (Earlier draft wrongly read 0.78 as
     the fused AUC and claimed a ranking win — corrected; no fused AUC exists.)
  2. **Key hyper-parameters unreported**: the per-bag instance count *n*
     (Table 1 uses it symbolically, never a value/distribution) and the
     threshold **β** are not given — so the bag→label decision cannot be
     rebuilt.
  3. **The α decision rule is self-contradictory** across the paper:
     (a) "more than *half* the instances exceed α → depression" (majority-
     positive), (b) "α = threshold at which a *single* instance determines the
     bag as *negative*" (single-negative), (c) Fig 4: "*at least one* exemplar
     > α" (any-one-positive). These three rules are mutually inconsistent.
  Consequence: **Multi-MTRB cannot be re-evaluated under our protocol**, so its
  0.88 can neither be confirmed nor refuted. Treat it as an *unverifiable,
  under-specified* number — flag in the writeup, do not adopt it as a firm bar.

Method: PDFs read 2026-06-13. Split/modality taken from each paper's own
Experiments/Results section (e.g. Milintsevich Table 2 "test set"; Mallol-Ragolta
Table 3 test UAR/F1; Xezonaki Table 5 "development set"; Alhanai "test set
annotations were not provided … evaluated on the development set"; Burdisso
"labels of the test set are not publicly available").
