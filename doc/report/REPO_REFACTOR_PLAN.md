# Repo refactor plan (deferred — run AFTER B6 + prob-levers finish)

DONE 2026-06-14 (safe, no import impact): deleted `results_old/` (297M), top-level
`scratch_*.py`, `parse_metrics.py`, `parse_v9d_log.py`, and 10 stale 185-era
`.sh` (run_tcmil_ablation{,2,2b,3}, rerun_kfold_*, run_thr_experiments,
run_tcmil_final, run_cv_oof, run_cv_lora). Active runners kept.

REMAINING (breaks imports → only when no job is running): move all code under
`src/` (decided) + delete legacy code + rewrite imports + update `.sh` paths.

## DELETE (legacy, not imported by the keep-set)
- `training/cv_ss_damil_r_v*.py` (~45), `training/cv_damil_*.py`,
  `training/train_damil_r*.py`, `training/train_ss_damil_r*.py`,
  `training/cv_ss_damil_r_v9d*.py`
- orphan models (only legacy trainers used them): `models/damil_cl.py`,
  `models/damil_x.py`, `models/damil_r_lora.py`, `models/ss_damil_r_v18.py`,
  `models/ss_damil_r_v20.py`, `models/ss_damil_r_v22.py`
- legacy entry/eval: `main.py`, `evaluation/evaluate_damil_r.py`,
  `extract_linguistic_features.py` (damil_x dep, now gone)
- stale docs: `README_LORA.md`, `README_config.md`, `readme_ss.md`,
  `results_mitigation_1.md`, `implementation_plan.md`
- REVIEW then likely delete: top-level `analysis/`, `plots/` (old), `dataset.py`
  only if eval_legacy is repointed (it imports `load_interviews_with_roles`).

## KEEP (paper + current pipeline) → move into `src/`
```
src/
├── core/            tcmil_data.py, preprocess_raw.py, dataset.py,
│                    models/{tcmil,damil_r,ss_damil_r}.py, utils/*
├── training/        train_tcmil_official, train_ncl_pair, train_multigran,
│                    finetune_tcmil, dapt_mlm
├── crossval/        cv_tcmil
├── ensemble/        combine_pw1_ensemble, ensemble_pair_search, ensemble_refine,
│                    combine_oof_ensemble, ensemble_tcmil_official, eval_b4,
│                    oof_threshold_official
├── interpretability/  interpret_tcmil
├── evaluation/      eval_legacy, mc_dropout_eval
├── statistics/      stats_ablation, stats_tcmil, summarize_finetune
└── plotting/        plot_faithfulness_scatter, plot_occlusion_compare,
                     plot_occlusion_saliency, build_ablation_table
```
Add `__init__.py` to every subpackage.

## IMPORT REWRITES (the breaking part)
Current cross-imports to remap everywhere:
- `from tcmil_data import` → `from src.core.tcmil_data import`
- `from preprocess_raw import` → `from src.core.preprocess_raw import`
- `from dataset import` → `from src.core.dataset import`
- `from models.X import` → `from src.core.models.X import`
- `from utils.X import` → `from src.core.utils.X import`
- `from training.train_tcmil_official import` → `from src.training.train_tcmil_official import`
- drop the `sys.path.append(parent)` hacks; run as `python -m src.<pkg>.<mod>`.

## .sh / .sbatch PATH UPDATES
`python training/X.py` → `python -m src.training.X` (or src/<pkg>/X.py) in:
`run_b6_ncl.sh`, `run_ensemble_roadmap.sh`, `run_prob_levers.sh`,
`run_tcmil_ablation_full189.sh`, `cluster/run_ft_*.sh`,
`cluster/ft_job*.sbatch`. Also `results/README.md` "Regenerate" commands.

## VERIFY after
Import-check every entry module; re-run one cheap analysis (build_ablation_table,
plot_faithfulness_scatter) to confirm paths resolve.

## STATUS: DONE (2026-06-14)
Executed. 62 legacy files deleted; all code under src/{core,training,crossval,ensemble,interpretability,evaluation,statistics,plotting}; imports rewritten (37/37 import OK); .sh/.sbatch/README repathed. Verified by import-check + smoke runs.
