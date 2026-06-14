# DAIC-WOZ text-only test-set — new literature candidates (2026-06-14)

## VERIFIED FROM PDFs (2026-06-14, paper/related_work/check/) — GROUND TRUTH
- **llm.pdf** = Merzougui, Dias, Pantin, Maurel, "Evaluating LLMs for Depression
  Symptom Estimation" (GREYC Caen, same group as Milintsevich/Agarwal). ✅
  **COMPARABLE**: text-only, official **test** (107/35/47), binary micro+macro F1.
  BUT all numbers are **PROMPTED LLMs (zero/few-shot)**, not trained models:
  Gemini-2.0-flash **0.84/0.85** (best, proprietary zero-shot), Mistral-7B 0.78,
  Llama-3.1-8B 0.69, Gemini-1.5-pro 0.64, DeepSeek-R1-8B 0.62. New top text-only
  test number but a DIFFERENT CATEGORY (closed/zero-shot, no training, no AUC, no
  bias control, prompt-unstable 0.49–0.84).
- **agent_mental.pdf** = AgentMental (Hu et al., HFUT). ❌ **NOT comparable** —
  "use the **development set** for evaluation" (NOT test). Exclude.
- **bert_ensembles.pdf** = Senn, Tlachac, Flores, Rundensteiner (WPI). ❌ **NOT
  comparable** — reformulates DAIC into 12 per-question "thematic datasets",
  reports mean F1 across them; no official-test subject-level macro-F1. Exclude.


Web search for text-only DAIC-WOZ papers reporting **test-set** performance,
beyond the 16 already in `paper/related_work/`. **All numbers below are
WEB-SOURCED and UNVERIFIED** — web summaries proved unreliable before (see
BASELINE-VERIFICATION.md). Download the PDFs and treat those as ground truth;
confirm modality + eval split + metric type per paper before citing.

Legend — "fit" = matches our comparison regime (text-only, official test,
binary PHQ-8≥10 classification, macro/micro-F1 or AUC).

## A. Likely comparable: text-only, official test, binary (PRIORITY to fetch)

| Paper | Authors | Venue/Year | Link | Web-claimed (UNVERIFIED) |
| --- | --- | --- | --- | --- |
| Evaluating LLMs for Depression Symptom Estimation | (author page: maurelf / GREYC) — confirm from PDF | AIME 2025 | https://maurelf.users.greyc.fr/docs/conferences/AIME_2025.pdf | text-only DAIC test; Gemini-2.0-Flash **macro-F1 0.84 / micro-F1 0.85** binary. (PDF didn't text-parse — verify authors+split) |
| AgentMental: Interactive Multi-Agent Framework for Explainable and Adaptive Mental Health Assessment | Jinpeng Hu, Ao Wang, Qianqian Xie, Hui Ma, Zhuo Li, Dan Guo | arXiv 2025 (Aug) | https://arxiv.org/abs/2508.11567 | text-based, DAIC-WOZ, "better than existing"; split + F1 not stated in abstract — verify |
| Ensembles of BERT for Depression Classification | ML Tlachac et al. (WPI) — confirm | (EnsembleBERT) | https://mltlachac.github.io/files/EnsembleBERT%20(4).pdf | text BERT ensemble on DAIC; split/F1 unparsed — verify whether test or dev |

## B. Text-only, test split, but REGRESSION / symptom-score (not binary — context only)

| Paper | Authors | Venue/Year | Link | Notes (UNVERIFIED) |
| --- | --- | --- | --- | --- |
| Interpretable depression assessment using a large language model | Jae-Joong Lee, Jihoon Han, Choong-Wan Woo | 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12885269/ | text-only; DAIC test n=46 + E-DAIC; **PHQ-8 regression (MAE/RMSE/R²), NO binary F1**. Comparable only if they add a binary cut |

## C. Validity / reproducibility (methodological — like Burdisso, for the validity discussion)

| Paper | Authors | Venue/Year | Link | Notes (UNVERIFIED) |
| --- | --- | --- | --- | --- |
| Most DAIC-WOZ Depression Classifiers Are Invalid, They Don't Learn Task-Specific Features: Preliminary Findings From a Large-Scale Reproducibility Study | (verify from PDF) | ICMI 2025 Companion | https://dl.acm.org/doi/10.1145/3747327.3763034 | reproducibility/validity study; argues many DAIC classifiers learn non-task features. Relevant to our bias-control narrative (ACM page 403'd — fetch PDF) |

## D. Multimodal or uncertain modality (download to confirm; EXCLUDE if not text-only)

| Paper | Authors | Venue/Year | Link | Web modality (UNVERIFIED) |
| --- | --- | --- | --- | --- |
| Optimizing depression detection in clinical doctor-patient interviews using a multi-instance learning framework | (login-walled) | Sci. Reports 2025 | https://www.nature.com/articles/s41598-025-90117-w | **MIL framework** — modality unknown (could be text MIL = very relevant to TC-MIL). Fetch to check text-only + split |
| Innovative Framework for Early Estimation of Mental Disorder Scores | Singh, Tiwari, Agarwal, Chandra, Sonbhadra, V. Singh | arXiv 2025 (2502.03965) | https://arxiv.org/abs/2502.03965 | multimodal (text+audio) LSTM/BiLSTM; acc 92% — likely EXCLUDE (multimodal) |
| LLMs for Depression Recognition in Spoken Language Integrating Psychological Knowledge | Yupei Li, Shuaijie Shao, Manuel Milling, Björn W. Schuller | arXiv 2025 (2505.22863) | https://arxiv.org/abs/2505.22863 | speech+text multimodal (Wav2Vec→LLM), MAE only — EXCLUDE (not text-only, regression) |
| Integrating LLMs into a Tri-Modal Architecture for Automated Depression Classification | (verify) | arXiv 2024 (2407.19340) | https://arxiv.org/abs/2407.19340 | tri-modal, **LOSO** (not official test); F1 85.95 LOSO — EXCLUDE (multimodal + not official test) |
| IntervoxNet (F1 0.90 on DAIC-WOZ) | (verify) | — | (search "IntervoxNet DAIC-WOZ") | F1 0.90 — modality unknown, suspiciously high (likely multimodal/dev) — verify |

## Already covered (in paper/related_work/, do NOT re-fetch)
milintsevich (0.739 macro test, our baseline) · mil_bio/Multi-MTRB · multilevel/MDSD-FGPL ·
mallolragolta/HCAN · prompt_bias/Burdisso · Alhanai · rohanian · 17_agent · 2024.clpsych ·
multimodal · multimodal2 · rf · svm · topic/Gong · topic2.

## Recommended fetch order
1. **AIME 2025 (greyc)** — strongest new text-only binary test claim (0.84/0.85).
2. **Nature MIL (s41598-025-90117-w)** — check if text-only MIL (direct method neighbour).
3. **ICMI 2025 validity study** — strengthens bias-control section.
4. **AgentMental**, **Ensembles-of-BERT** — confirm split (test vs dev) before use.
