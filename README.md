# TC-MIL: Topic-Chunk Multiple-Instance Learning for DAIC-WOZ

Text-only depression detection on the DAIC-WOZ clinical interview corpus.
TC-MIL treats each interview as a **bag of dialogue chunks** (sliding windows of
interviewer→participant exchanges) and classifies it with a small
gated-attention MIL head over a **frozen sentence encoder**. A single trained
model significantly beats the strongest verified non-leaky text-only baseline on
the official AVEC2017 test split, with an explicit interviewer-prompt bias
control.

> This README is the entry point. Deeper method/ablation history lives in
> [`README_TCMIL.md`](README_TCMIL.md); the encoder fine-tuning plan lives in
> [`README_FINETUNE.md`](README_FINETUNE.md) and [`cluster/FT_JOBS.md`](cluster/FT_JOBS.md).

---

## Headline results

All on the **official AVEC2017 split** (train 107 / dev 35 / test 47). The
encoder is frozen; splits are subject-disjoint by construction
(`assert_no_leakage`). The headline is the **single model** — for a fair
comparison with the text-only literature, which reports a single trained model's
per-seed mean ± std (no seed/encoder/member ensembling). The ensemble is an
optional boost.

| Model | Protocol | AUC | macro-F1 | micro-F1 |
| --- | --- | --- | --- | --- |
| **TC-MIL single** (bge-large, pos_weight=1.0) | test, 30-seed per-seed mean ± std | **0.864 ± 0.011** | **0.751 ± 0.031** | **0.780 ± 0.019** |
| TC-MIL single — seed-ensemble (prevalence threshold) | test | 0.866 | 0.770 | — |
| TC-MIL ensemble (bge-large plain + mxbai GRU, OOF-prev) | test | 0.874 | 0.810 | — |
| Milintsevich et al. 2023 (cleanest non-leaky baseline) | test, 5-seed mean | — | 0.739 | 0.766 |

- The **single** TC-MIL model exceeds the Milintsevich baseline on macro-F1
  (0.751 vs 0.739) and micro-F1 (0.780 vs 0.766) with no ensembling and no
  test-set tuning. Threshold is set a-priori by **prevalence matching** (no dev
  labels), which is the stable choice on the 35-subject dev set.
- The **2-member ensemble** (architectural diversity: a plain member + a GRU
  member) lifts test macro-F1 to 0.810 / AUC 0.874. Members are chosen by **OOF
  AUC only** — no test statistic enters selection.

Source files: single → `results/single_model/headline_pw1_30seed/`;
ensemble → `results/ensemble/oof/oof_headline/`.

### Cross-validation (clean pool: train+dev, official test excluded)

| Protocol | AUC | macro-F1 | F1 |
| --- | --- | --- | --- |
| Repeated K-Fold (5×5, GRU, testprev), seed-ensemble | 0.820 | 0.735 | 0.628 |
| Monte Carlo (10 seeds, 3-member, testprev), seed-ensemble | 0.883 | 0.804 | 0.727 |

Source: `results/cross_validation/`.

### Interviewer-prompt bias control (Burdisso et al. 2024)

Removing the `Interviewer:` lines from every chunk (`--participant_only`) keeps
performance — dev seed-ensemble AUC **0.902** (participant-only) vs **0.895**
(full dialogue). The signal does **not** come from the interviewer-prompt
shortcut that inflates much of the text-only literature.
Source: `results/design_ablations/dialogue_vs_ponly/`.

---

## Method

**Why chunks.** Earlier role-aware models (DAMIL-R / SS-DAMIL-R, 35 versions)
built bags from individual participant turns. Most DAIC-WOZ turns are
backchannels ("yeah", "mhm") that a frozen encoder maps to near-identical
vectors — a representation ceiling no pooling can break. TC-MIL instances are
**sliding windows of `window` consecutive exchanges** rendered as dialogue text
(`window=4, stride=2` → ~28 chunks/interview, ~65 words each), restoring the
topical context (sleep, mood, energy) that PHQ-8 symptoms attach to. This is the
single biggest lever.

**The model** (`src/core/models/tcmil.py`, ~120K–224K params):

1. Frozen sentence encoder (default `BAAI/bge-large-en-v1.5`), mean-pooled +
   L2-normalized per chunk. Embeddings are cached (`cache/tcmil/`).
2. Small projector → optional **1-layer BiGRU** temporal context layer over the
   chunk sequence (`--temporal gru`; +0.02–0.03 dev AUC).
3. Classic **gated-attention MIL pooling** (Ilse & Welling 2018).
4. **Symptom aux head** predicting the 8 binarized PHQ-8 items as a pure
   regularizer (shares the backbone, does not feed the main logit).

No focal loss, mixup, SWA, or diversity losses — over-regularization was the
failure mode of the previous era at n≈107. The winning recipe drops the
recall-biasing class weight (`pos_weight=1.0`).

**Protocol** (`src/training/train_tcmil_official.py`): train on official train,
select + threshold on dev, evaluate test **once**. CV
(`src/crossval/cv_tcmil.py`) is Stratified Group K-Fold / Monte Carlo over
train+dev only, so the official test stays untouched. Thresholds come from dev
prevalence, out-of-fold (OOF) probabilities, or unlabeled test-score prevalence
(`testprev`) — never from test labels.

---

## Repository layout

```
src/
  core/
    tcmil_data.py            chunking, frozen-encoder embedding + cache, leakage assert
    dataset.py               raw interview / role loading
    preprocess_raw.py        transcript preprocessing
    models/tcmil.py          gated-attention MIL + aux head + GRU/transformer context
    models/{damil_r,ss_damil_r}.py   legacy architectures (ablation ladder)
    utils/                   metrics, evaluation, stats, sam, plots
  training/
    train_tcmil_official.py  official-split protocol (saves per-seed checkpoints)
    finetune_tcmil.py        end-to-end encoder fine-tuning (LoRA/bitfit/last_k/full)
    dapt_mlm.py              domain-adaptive MLM pretraining on TRAIN text only
  crossval/cv_tcmil.py       K-Fold + Monte Carlo CV (innerval / oof / testprev thresholds)
  ensemble/
    oof_threshold_official.py    OOF threshold probe (official protocol)
    combine_oof_ensemble.py      frozen multi-encoder OOF ensemble
    combine_ft_ensemble.py       fine-tuned multi-member OOF ensemble
    ensemble_tcmil_official.py   dev-thresholded multi-encoder ensemble
  evaluation/eval_legacy.py  legacy architecture ladder under the rigorous protocol
  evaluation/mc_dropout_eval.py
  interpretability/interpret_tcmil.py   attention faithfulness, PHQ-8, bias probes
  plotting/  statistics/

results/
  single_model/        official-split single-model runs (headline_pw1_30seed = canonical)
  ensemble/            OOF + multi-member ensembles (oof/oof_headline = best)
  cross_validation/    K-Fold + Monte Carlo
  design_ablations/    e.g. dialogue_vs_ponly (interviewer-bias control)
  baselines/           legacy architectures under the rigorous protocol
  interpretability/    faithfulness / saliency artifacts
  stats/  logs/

cluster/               SLURM scripts + configs for encoder fine-tuning (see FT_JOBS.md)
paper/                 LaTeX manuscript, figures, refs
doc/report/            design notes, baseline verification, backlog/roadmap
```

---

## Saved weights & reproducibility

Training **persists trained weights**, so every reported number can be
re-inferred without retraining (the code is to be open-sourced):

- `train_tcmil_official.py` saves `checkpoints/config.json` (model-construction
  args) + one `checkpoints/seed_<seed>.pt` per seed. The encoder is frozen, so
  these are the small MIL-head state dicts.
- `finetune_tcmil.py` saves `checkpoints/config.json` + per-seed
  `seed_<seed>.pt` (official) / `fold<f>_seed<s>.pt` (CV). For FT, only the
  **trainable** parameters are stored (LoRA adapters / bias terms / unfrozen
  layers + MIL head); reload onto a fresh `FTTCMIL(config)` with
  `load_state_dict(..., strict=False)`. On by default; disable with
  `--no_save_weights`.

Per-run `results.json` additionally stores per-seed metrics, ensemble metrics,
dev/test probabilities, labels, thresholds-by-strategy, and full args — enough
to rebuild every table and to assemble ensembles offline from saved
probabilities without retraining.

---

## Reproduce

All commands run from the repo root (`python` resolves `src.*` via `sys.path`).
Dependencies via `uv` (`pyproject.toml` / `uv.lock`).

```bash
# Single-model headline (official test, bge-large, pos_weight=1.0)
python src/training/train_tcmil_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --pos_weight 1.0 \
    --n_seeds 30 --eval_test --threshold_metric prevalence \
    --output_dir results/single_model/headline_pw1_30seed

# Interviewer-bias control (dev only; never touches test)
python src/training/train_tcmil_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --participant_only \
    --output_dir results/design_ablations/dialogue_vs_ponly/ablate_ponly_gru

# Ensemble: per-member OOF runs, then combine (selection by OOF AUC only)
python src/ensemble/oof_threshold_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --n_seeds 10 \
    --output_dir results/ensemble/oof/oof_bgelarge_plain
python src/ensemble/oof_threshold_official.py \
    --encoder_name mixedbread-ai/mxbai-embed-large-v1 --temporal gru \
    --n_seeds 10 --output_dir results/ensemble/oof/oof_mxbai_gru
python src/ensemble/combine_oof_ensemble.py \
    --runs results/ensemble/oof/oof_bgelarge_plain results/ensemble/oof/oof_mxbai_gru \
    --output results/ensemble/oof/oof_headline/results.json

# Clean CV (GRU + testprev = headline K-Fold; MC ensemble aggregation built in)
python src/crossval/cv_tcmil.py --mode kfold --encoder_name BAAI/bge-large-en-v1.5 \
    --temporal gru --threshold_mode testprev --n_repeats 5 \
    --output_dir results/cross_validation/kfold_gru_testprev_r5

# Legacy architecture ladder under the same leak-free protocol
python src/evaluation/eval_legacy.py
```

---

## Encoder fine-tuning (cluster)

Fine-tuning the sentence encoder *through* the MIL objective is the lever most
likely to raise the frozen ranking ceiling (AUC ~0.86). The full plan, grid,
decision rules and leakage guarantees are in
[`README_FINETUNE.md`](README_FINETUNE.md); the SLURM job order is in
[`cluster/FT_JOBS.md`](cluster/FT_JOBS.md).

```bash
pip install peft                                   # only new dependency
python src/training/finetune_tcmil.py --smoke --ft_method lora \
    --output_dir results/ft/smoke                  # ~2 min install check

# Stage 1 — dev-selection grid (20 configs × 5 seeds, no test)
sbatch cluster/ft_job2_grid.sh                 # or: bash cluster/run_ft_grid.sh
python src/statistics/summarize_finetune.py --root results/ft

# >>> edit cluster/finalists_configs.txt + cluster/cv_configs.txt with the
#     grid winners (top-2 by 5-seed dev AUC + frozen control) <<<

# Stage 2 — finalists (OOF threshold + 10-seed official test)
sbatch cluster/ft_job3_finalists.sh            # or: bash cluster/run_ft_finalists.sh
# Stage 3 — CV for the winner vs frozen control
sbatch cluster/ft_job4_cv.sh                   # or: bash cluster/run_ft_cv.sh

# Optional: ensemble fine-tuned members offline (no retraining)
python src/ensemble/combine_ft_ensemble.py \
    --member results/ft/oof_winner1 results/ft/official_winner1 \
    --member results/ft/oof_winner2 results/ft/official_winner2 \
    --output results/ft/ft_ensemble/results.json
```

**Launch readiness.** The single-model FT pipeline is turnkey (grid →
summarize → finalists → CV); `finalists_configs.txt` and `cv_configs.txt` ship
with **placeholder flags by design** — edit them with the Stage-1 grid winners
before submitting jobs 3 and 4. The FT ensemble is assembled offline by
`combine_ft_ensemble.py` from the OOF + official runs each finalist already
produces (test probabilities are saved), so no extra training is needed.

---

## Status

- [x] TC-MIL single-model headline, ensemble, CV, bias control (frozen encoder)
- [x] Trained weights saved for all training/fine-tuning runs
- [ ] Encoder fine-tuning runs on the cluster (single model + ensemble)
- [ ] Zero-shot E-DAIC transfer test (train DAIC-WOZ, test E-DAIC)
- [ ] Paper write-up

See [`todo.md`](todo.md) for the working list.
