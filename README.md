# TC-MIL: Topic-Chunk Multiple-Instance Learning for DAIC-WOZ

Text-only depression detection on the DAIC-WOZ clinical-interview corpus. Each
interview is a **bag of dialogue chunks** (sliding windows of
interviewer→participant exchanges) classified by a small gated-attention MIL head
over a **frozen sentence encoder**.

---

## 1. Requirements

- **Python ≥ 3.11** and [**uv**](https://docs.astral.sh/uv/) (dependency manager).
- A **CUDA GPU** is strongly recommended for training/fine-tuning (CPU works for
  inference and small runs, just slowly).
- Access to the **DAIC-WOZ** corpus (and optionally **E-DAIC**) under their EULA —
  needed to download both raw transcripts and label splits.

All Python deps (torch, transformers, peft, scikit-learn, …) are pinned in
`pyproject.toml` / `uv.lock`.

---

## 2. Setup

```bash
git clone <repo> && cd tcmil
uv sync                      # creates .venv/ and installs everything
```

Run commands either with `uv run python …` or by activating the venv
(`source .venv/bin/activate`, then `python …`). Examples below use `uv run`.
All commands run **from the repo root** (imports resolve `src.*` via the repo
root on `sys.path`).

---

## 3. Get the data

**DAIC-WOZ and E-DAIC are access-controlled and may not be redistributed.** You
must request access and agree to the data-use terms; once approved you receive
the base URL for the participant archives.

> **Request access:** [link](https://dcapswoz.ict.usc.edu)

This repo ships **neither transcripts nor labels** — both come from your
approved download. Two manual steps:

**(a) Label files — place by hand.** Download the split CSVs from the approved
source and drop them in these exact paths/names (the loaders look for them
literally):

```
data/daic-woz/labels/
    train_split_Depression_AVEC2017.csv
    dev_split_Depression_AVEC2017.csv
    full_test_split.csv
data/e-daic/labels/                       # only for the zero-shot experiment (4e)
    train_split.csv
    dev_split.csv
    test_split.csv
    Detailed_PHQ8_Labels.csv
```

**(b) Transcripts — scripted.** Once labels are in place, supply your base URL to
the download scripts (pass `--base-url <URL>` per run, or paste it once into
`PLACEHOLDER_BASE_URL` at the top of each script). They read participant IDs from
your labels and fetch **text only** (audio/video discarded), and are resumable
(existing transcripts skipped). Without a URL they exit with an error.

```bash
# DAIC-WOZ (the main corpus)
uv run python -m src.core.download_daicwoz --base-url <YOUR_DAICWOZ_URL>                 # all splits
uv run python -m src.core.download_daicwoz --base-url <YOUR_DAICWOZ_URL> --splits test   # one split
uv run python -m src.core.download_daicwoz --base-url <YOUR_DAICWOZ_URL> --limit 5       # smoke test

# E-DAIC (only needed for the zero-shot transfer experiment, step 4e)
uv run python -m src.core.download_edaic --base-url <YOUR_EDAIC_URL>
```

Resulting layout (transcripts land in `raw/`; preprocessing into dialogue
chunks happens **in memory** at train time and embeddings are cached, so there
is no separate preprocessing step to run):

```
data/daic-woz/
  labels/     AVEC2017 split CSVs   (placed by hand, step a)
  raw/        {ID}_TRANSCRIPT.csv   (downloaded, step b)
data/e-daic/
  labels/     AVEC2019 split CSVs   (placed by hand, step a)
  raw/        {ID}_TRANSCRIPT.csv   (downloaded, converted to DAIC-WOZ schema)
```

---

## 4. Run the experiments

Each step is independent; run only what you need. Outputs go under `results/`
and trained weights under `checkpoints/` (see [6](#6-outputs-weights-reproducibility)). The first run of any encoder
embeds + caches chunks to `cache/tcmil/` (slow once, fast after).

### 4a. Single-model headline (the main result)

```bash
uv run python src/training/train_tcmil_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --pos_weight 1.0 \
    --n_seeds 30 --eval_test --threshold_metric prevalence \
    --output_dir results/single_model/headline_pw1_30seed
```

Trains the official split (train 107 / dev 35 / test 47), 30 seeds, evaluates the
test split **once** at an a-priori prevalence threshold. **Expect** ROC-AUC
≈ 0.86 (per-seed mean). Per-seed metrics, probabilities, thresholds, and args
land in `results/.../results.json`.

### 4b. Cross-validation (test split untouched)

```bash
# Repeated K-Fold (5×5) on the train+dev pool
uv run python src/crossval/cv_tcmil.py --mode kfold \
    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru \
    --threshold_mode testprev --n_repeats 5 \
    --output_dir results/cross_validation/kfold_gru_testprev_r5

# Monte-Carlo CV
uv run python src/crossval/cv_tcmil.py --mode mc \
    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru \
    --output_dir results/cross_validation/mc_gru_single
```

### 4c. Interviewer-prompt bias control (dev only)

```bash
uv run python src/training/train_tcmil_official.py \
    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --participant_only \
    --output_dir results/design_ablations/dialogue_vs_ponly/ablate_ponly_gru
```

`--participant_only` strips the `Interviewer:` lines — checks the model is not
exploiting the interviewer-prompt shortcut. Never touches the test split.

### 4d. Design ablations + legacy architecture ladder

```bash
bash scripts/experiments/design_ablations.sh   # encoder / temporal / window / reg sweeps (dev, 5-seed)

# Legacy architecture ladder (baselines): dialogue-mean → flat-MIL → DAMIL-R → SS-DAMIL-R.
# Run at TC-MIL's matched CV budget so the paired tests align run-for-run.
uv run python src/evaluation/eval_legacy.py --data_dir data/daic-woz \
    --protocols official kfold --n_seeds 30 --n_repeats 5 --cv_seeds 5 --n_splits 5 \
    --output_root results/baselines/legacy_aligned
# Monte-Carlo baselines (5 splits × 10 seeds), aligned to TC-MIL's mc_pw1:
uv run python src/evaluation/eval_legacy.py --data_dir data/daic-woz \
    --protocols mc --cv_seeds 10 --n_splits 5 \
    --output_root results/cross_validation/legacy_mc_aligned
```

`scripts/experiments/pw_sweep30.sh` and `scripts/experiments/prob_levers.sh` run
the pos-weight sweep and the checkpointed-headline + MC-dropout + multi-granularity
variants. All are standalone and resumable (finished runs are skipped). Regenerate
the significance tables (ladder + CV, all metrics) with
`uv run python -m src.statistics.stats_ablation --protocols kfold mc --independent`.

### 4e. Zero-shot E-DAIC transfer

Score a DAIC-WOZ-trained checkpoint on E-DAIC with **no retraining** (download
E-DAIC first, step 3):

```bash
uv run python -m src.training.eval_edaic_zeroshot \
    --checkpoint_dir results/single_model/headline_pw1_30seed/checkpoints \
    --edaic_dir data/e-daic \
    --output_dir results/zeroshot/edaic
```

Pass the same `--window/--stride/--gap_merge` used to train the source model so
the only shift measured is the domain shift.

### 4f. Encoder fine-tuning (SLURM cluster)

Tests whether fine-tuning the encoder *through* the MIL objective beats the
frozen baseline. Staged so the test split is spent once. Scripts +
editable config files are in `scripts/finetune/` (see [5](#5-scripts)).

```bash
# Install check (~2 min on GPU)
uv run python src/training/finetune_tcmil.py --smoke --ft_method lora \
    --output_dir results/ft/smoke

# (optional) DAPT checkpoint, only for the two dapt_* grid configs
sbatch scripts/finetune/job1_dapt.sh

# Stage 1 — dev-selection grid (20 configs × 5 seeds, NO test)
sbatch scripts/finetune/job2_grid.sh        # or: bash scripts/finetune/run_grid.sh
uv run python src/statistics/summarize_finetune.py --root results/ft

#   >>> the summary prints the pre-registered finalists. Edit
#       scripts/finetune/configs/finalists_configs.txt and cv_configs.txt
#       with the winners (+ frozen control) before the next stage. <<<

# Stage 2 — finalists: leakage-free OOF threshold + 30-seed official test
sbatch scripts/finetune/job3_finalists.sh   # or: bash scripts/finetune/run_finalists.sh

# Stage 3 — cross-validation: winner vs frozen control
sbatch scripts/finetune/job4_cv.sh          # or: bash scripts/finetune/run_cv.sh
```

Before submitting: adjust the `#SBATCH --partition/--account` lines in each
`job*.sh` to your cluster. Jobs are **resumable** — array tasks self-throttle to
4 concurrent (`%4`), finished protocol calls and per-seed units are cached, so on
a wall-time kill just `sbatch` the **same** script again until it finishes. Set
each job's `--array` to its config line count (job 3 = #finalists + 1, job 4 =
#cv lines). Jobs 3 and 4 are independent and can run together.

---

## 5. Scripts

| Path | What |
| --- | --- |
| `scripts/experiments/design_ablations.sh` | design ablations (encoder/temporal/window/reg), dev 5-seed |
| `scripts/experiments/pw_sweep30.sh` | pos-weight sweep, 30-seed official |
| `scripts/experiments/prob_levers.sh` | checkpointed headline + MC-dropout + multi-granularity |
| `scripts/finetune/job1_dapt.sh` | SLURM: DAPT MLM pretrain → `checkpoints/dapt_bge_large` |
| `scripts/finetune/job2_grid.sh` | SLURM: Stage-1 grid (array of 20) |
| `scripts/finetune/job3_finalists.sh` | SLURM: Stage-2 OOF threshold + 30-seed official test |
| `scripts/finetune/job4_cv.sh` | SLURM: Stage-3 K-Fold(+repeat) + MC, winner vs frozen |
| `scripts/finetune/run_{grid,finalists,cv}.sh` | the bodies the SLURM jobs call (also runnable directly) |
| `scripts/finetune/slurm_grid.sh` | plain (non-dependency) SLURM wrapper for the grid |
| `scripts/finetune/configs/*.txt` | one `name\|flags` per line; **edit finalists/cv after the grid** |

---

## 6. Outputs, weights, reproducibility

Every run writes `results/<dir>/results.json` with per-seed metrics,
seed-ensemble metrics, dev/test probabilities + labels, thresholds-by-strategy,
and the full args — enough to rebuild every table without retraining.

Trained weights are **saved by default** (disable with `--no_save_weights`),
under `<output_dir>/checkpoints/`:

- `train_tcmil_official.py` → `config.json` + one `seed_<seed>.pt` per seed
  (small frozen-head state dicts).
- `finetune_tcmil.py` → `config.json` + per-seed `seed_<seed>.pt` (official) /
  `fold<f>_seed<s>.pt` (CV); **only trainable** params (LoRA adapters / bias /
  unfrozen layers + MIL head). Reload onto a fresh `FTTCMIL(config)` with
  `load_state_dict(..., strict=False)`.

[This](https://drive.google.com/drive/folders/1IWKe75-f6cKfh6agss5QC1OMXZC0bYcr?usp=sharing) folder holds the single-model track
(headline + explored levers) with a reload example — see `README.md` inside it.

**Leakage guarantees:** splits are subject-disjoint by construction
(`assert_no_leakage`); CV fine-tunes the encoder **inside each fold** only;
thresholds come from dev prevalence, OOF probabilities, or unlabeled
test-score prevalence (`testprev`) — never from test labels.

---

## 7. Repository layout

```
src/
  core/            data loading, chunking + embedding cache, models, dataset download
  training/        train_tcmil_official, finetune_tcmil, dapt_mlm, train_multigran, eval_edaic_zeroshot
  crossval/        cv_tcmil, oof_threshold_official
  evaluation/      eval_legacy, mc_dropout_eval
  interpretability/ interpret_tcmil (attention faithfulness, PHQ-8, bias probes)
  statistics/      stats_tcmil, summarize_finetune, oof_ft_frozen_compare, …
  plotting/        figure builders
scripts/           runnable experiment + cluster scripts (see 5. Scripts)
data/              daic-woz/ + e-daic/ (user-provided labels + downloaded transcripts; git-ignored)
results/           run outputs (results.json per run)
checkpoints/       saved weights
paper/             LaTeX manuscript, figures, refs
doc/report/        design notes, baseline verification, backlog/roadmap (history)
```
