# TC-MIL Encoder Fine-Tuning — Cluster Experiment Plan

Goal: test whether fine-tuning the sentence encoder through the MIL objective
can raise the frozen-encoder **ranking** ceiling (AUC ~0.86 frozen) — the
single lever most likely to reach the fusion baselines (MDSD-FGPL macro 0.874;
Multi-MTRB unverifiable/irreproducible, see `doc/report/BASELINE-VERIFICATION.md`).

**Headline = single model** (the project pivoted to single-model focus; see
`doc/report/BACKLOG.md`). The winning frozen recipe is **bge-large + GRU +
`pos_weight = 1.0`** (dropping the recall-biasing class weight lifted macro-F1
0.751 → 0.774, significant vs Milintsevich — `results/full189/SINGLE_MODEL_IMPROVEMENTS.md`).
FT defaults to `pos_weight = 1.0`; the frozen control must use it too.

Current **single-model** frozen references (the bar FT must clear):

| Benchmark (frozen, single model, pos_weight=1.0) | AUC | macro-F1 | micro-F1 |
| --- | --- | --- | --- |
| Official test (bge-large, 30-seed per-seed) | **0.864 ± 0.011** | **0.774 ± 0.029** | 0.814 ± 0.024 |
| K-Fold per-run (5×5, GRU, testprev) | 0.799 | 0.706 | — |
| Milintsevich baseline (to beat) | — | 0.739 | 0.766 |
| MDSD-FGPL (fusion target) | — | 0.874 | — |

## Why this grid (and not more)

107 training subjects / ~3K chunks vs a 335M-parameter encoder: the central
risk is destroying pretrained knowledge, not underfitting. Every grid point
is a different stake on that axis — bias-only (BitFit), low-rank adapters
(LoRA r8→r32, attention-only vs +FFN), partial unfreezing (last 2/4 layers
with layer-wise lr decay 0.8), and full FT only at very low lr (5e-6/1e-5)
as the boundary probe. Built-in protections in `training/finetune_tcmil.py`:

- **Head warmup** (`--head_warmup_epochs 3`): encoder lr held at 0 until the
  randomly-initialized MIL head stops sending garbage gradients.
- **Early stopping** on dev (official) / inner-val (CV) AUC, patience 8.
- **Linear warmup/decay**, grad clipping 1.0, bf16 autocast,
  gradient checkpointing.
- Frozen control runs in the same script — any environment/protocol drift
  shows up there first.

Deliberately excluded (near-guaranteed failures at n=107): training
embeddings layer alone, lr ≥ 1e-4 for non-adapter weights, contrastive
objectives on bag pairs (≈5K pairs, all from 107 subjects), prompt tuning
(needs thousands of steps to converge), and any method that fine-tunes on
pooled CV data (leakage).

## Leakage rules (enforced in code, do not relax)

1. Official: encoder sees TRAIN text only; dev is used for early stopping
   and selection; test evaluated once per finalist with an OOF threshold.
2. CV (`--protocol kfold|mc`): the encoder is re-fine-tuned from the
   pretrained checkpoint **inside every fold** on that fold's training
   subjects only. Never fine-tune once on the pool and then cross-validate.
3. DAPT (`training/dapt_mlm.py`): MLM on the official TRAIN split text only.
4. Thresholds: dev-prevalence, OOF (`--protocol export_oof` →
   `--threshold_file`), or testprev (predicted-positive-rate matching on
   unlabeled fold-test scores). Never tuned on test labels.

## How to run (cluster)

Environment notes (current project state, 2026-06-13):
- **`pos_weight = 1.0` is the default** in `finetune_tcmil.py` (the winning
  single-model recipe). Keep it; do not revert to auto class weighting.
- **transformers 5.x** is in use. Load richer encoders **natively** (no
  `trust_remote_code`) — e.g. gte-Qwen2-1.5B loads as a base `Qwen2Model`;
  the model's *remote* tokenizer code is incompatible with transformers 5.
- **LoRA targets auto-detect** the attention module names (BERT
  query/key/value *or* causal-LM q_proj/k_proj/v_proj), so the grid works for
  bge/mxbai *and* Qwen2/Mistral-class encoders without edits.
- **micro-F1** is now emitted by `compute_metrics`; report it alongside macro.

```bash
# 0. Setup
pip install peft                       # only new dependency
python training/finetune_tcmil.py --smoke --ft_method lora \
    --output_dir results/ft/smoke      # install check, ~2 min on GPU

# 1. DAPT checkpoint (needed by 2 grid lines)
python training/dapt_mlm.py --output_dir checkpoints/dapt_bge_large

# 2. Stage 1 — dev-selection grid (20 configs x 5 seeds, no test)
sbatch cluster/slurm_ft_grid.sbatch    # or: bash cluster/run_ft_grid.sh
python training/summarize_finetune.py --root results/ft   # grid table

# 3. Stage 2 — finalists (EDIT the FINALISTS array first, see decision rules)
bash cluster/run_ft_finalists.sh

# 4. Stage 3 — CV for the single winner (EDIT WINNER_FLAGS first)
bash cluster/run_ft_cv.sh
```

Estimated cost on one A100: grid ≈ 15–25 GPU-h (10–25 min/seed, early
stopping usually fires by epoch 15–20); finalists ≈ 8 GPU-h; CV ≈ 15 GPU-h.
Lines of `cluster/ft_grid_configs.txt` are independent → SLURM array.

## Pre-registered decision rules (set before seeing results)

All gates are on the **single model** (per-seed mean ± std), not ensembles.

1. **Sanity gate**: `grid_frozen_ctrl` (frozen, pos_weight=1.0) must land at
   per-seed dev AUC ≈ 0.90 ± 0.02. Outside that → environment/protocol bug;
   stop before reading any other number.
2. **Forgetting flag**: any config with per-seed dev AUC mean < 0.80 is dead
   (catastrophic forgetting); exclude.
3. **Stage-1 winner(s)**: top-2 configs by 5-seed **per-seed dev AUC mean**;
   ties broken by lower std. Adoption requires the FT dev AUC to clear the
   frozen control from the **same run** by more than its seed noise
   (≈ +0.015). Below that → "frozen ceiling stands", stop after stage 1.
4. **Stage 2**: per finalist, 10-seed official, ONE test evaluation, a-priori
   prevalence threshold. Headline metric = **per-seed test macro-F1 / AUC**.
   Compare against the frozen single-model bar: **AUC 0.864 / macro-F1 0.774**
   (and the targets: Milintsevich 0.739, MDSD-FGPL 0.874). Win = FT single
   significantly > frozen single on macro-F1 (one-sample t over seeds).
5. **Stage 3**: winner vs frozen control on K-Fold (repeats × 5 folds) and MC,
   testprev thresholds, **per-run** (single-model) metrics. Improvement claim
   requires FT single-model AUC/macro-F1 to clear the same-script frozen
   control on both protocols.

## Artifacts to bring back for interpretation

- `results/ft/**/results.json` + `run.log` (everything below derives from these)
- `results/ft/oof_*/oof_thresholds.json`
- the `summarize_finetune.py` table
- `checkpoints/dapt_bge_large/dapt.log` (MLM loss curve)

Per-run `results.json` contains: per-seed dev metrics, dev-ensemble metrics,
dev/test probabilities and labels, thresholds by strategy, full args. That is
sufficient to (a) rebuild every table, (b) run Wilcoxon between configs on
matched seeds (`training/stats_tcmil.py wilcoxon`), (c) bootstrap test CIs
(`stats_tcmil.py bootstrap`), and (d) build mixed frozen+FT ensembles
offline from the saved probabilities (the FT analogue of
`combine_oof_ensemble.py` — test probs are saved, so member averaging needs
no re-training).

## Failure signatures to watch in run.log

- sel AUC collapses toward 0.5 within 2–3 epochs after warmup → encoder lr
  too high for that method; the lower-lr sibling config is the answer.
- sel AUC peaks at epoch 1–2 then degrades monotonically → forgetting;
  early stopping already saved the best state, but treat the config as dead
  per rule 2 if the peak is < 0.80.
- frozen control deviating from 0.90 → environment problem
  (tokenizer/version drift), nothing else is interpretable.
- bf16 NaNs (loss=nan in log): rerun the config with
  `--no_grad_checkpointing` first; if it persists, add `--accum 16`.
