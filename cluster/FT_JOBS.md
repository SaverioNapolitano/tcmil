# Fine-tuning on the cluster — job order & parallelism

Four numbered SLURM scripts. Just `sbatch` them in order. Arrays are throttled
to **4 concurrent tasks** (`%4`) to respect the 4-parallel-job cap. Before the
first submit: `pip install peft` and adjust the `##SBATCH --partition/--account`
lines in each `.sh` to your cluster.

## The jobs

| # | file | what | array | needs | edit first? |
|---|------|------|-------|-------|-------------|
| 1 | `ft_job1_dapt.sh` | DAPT MLM pretrain → `checkpoints/dapt_bge_large` | — | — | no (optional) |
| 2 | `ft_job2_grid.sh` | Stage-1 grid, 20 configs, dev-only 5-seed | `1-20%4` | job 1 (for DAPT lines 19-20) | no |
| 3 | `ft_job3_finalists.sh` | Stage-2: OOF threshold + 30-seed official **test** | `1-3%4` | job 2 + edit configs | **yes** |
| 4 | `ft_job4_cv.sh` | Stage-3: K-Fold(+repeat)+MC, winner vs frozen | `1-2%4` | job 2 + edit configs | **yes** |

Configs (one `name|flags` per line):
- job 2 → `cluster/ft_grid_configs.txt` (ready; 20 lines)
- job 3 → `cluster/finalists_configs.txt` (**edit after job 2**: configs within a dev seed-noise band of the best 5-seed dev AUC, capped at 4, + frozen — `summarize_finetune.py` prints the selection; set the job-3 array to #finalists+1)
- job 4 → `cluster/cv_configs.txt` (**edit after job 2**: the single winner + frozen)

## Run order (copy-paste)

```bash
# --- optional DAPT arm (skip if not using dapt_* grid configs) ---
d=$(sbatch --parsable cluster/ft_job1_dapt.sh)

# --- Stage 1: grid (depends on DAPT only for lines 19-20) ---
g=$(sbatch --parsable --dependency=afterok:$d cluster/ft_job2_grid.sh)
#   no DAPT? instead:  g=$(sbatch --parsable cluster/ft_job2_grid.sh)   # and set --array=1-18%4

# >>> when job 2 finishes: run `python training/summarize_finetune.py` and
#     edit finalists_configs.txt + cv_configs.txt with the winners <<<

# --- Stage 2 + Stage 3 (independent of each other, both need the grid) ---
sbatch --dependency=afterok:$g cluster/ft_job3_finalists.sh
sbatch --dependency=afterok:$g cluster/ft_job4_cv.sh
```

Or, fully manual (no dependencies): `sbatch job1` → wait → `sbatch job2` → wait,
summarize, edit configs → `sbatch job3` and `sbatch job4`.

## Parallelism summary
- **Sequential:** job1 → job2 → {job3, job4}. (3 and 4 both need 2's results.)
- **Parallel:** job 3 and job 4 can run together. Each array self-throttles to 4.
  If you submit both at once and want to stay well under the cap, change their
  `%4` to `%2`.
- Within a job, the array runs ≤4 tasks at a time automatically.

## Resuming after a wall-time kill (all jobs)
`--time` is **per array task**, not for the whole array. The 14h30 account cap
means jobs 3 and 4 likely need >1 submit, so resume is two-level and **on by
default** — just resubmit the **same** `.sh` and it continues near where it
stopped:

1. **Per protocol call** (the `uv run ...` lines in `run_ft_*.sh`): skipped if
   its `results/ft/<dir>/results.json` already exists (written only on
   completion). So a finished `export_oof`/`kfold`/`mc`/`official` call returns
   instantly on resubmit.
2. **Per seed / fold-seed** (inside one call): every finished training unit is
   cached to `results/ft/<dir>/seed_cache/` (`seed_<n>.json` for `official`,
   `fold<f>_seed<n>.json` for cv) the moment it completes — predictions +
   metrics, written atomically so a kill mid-write can't corrupt it. On resubmit
   those units load from cache (no retrain); only the missing ones run. CV
   splits are seeded independently of which seeds ran, so this is exact.

So for a killed job 3 / job 4: just `sbatch` the **same** script again (and
again) until it finishes. The cache is keyed by the run config — if you change a
hyperparam in the same output dir, stale units are detected and recomputed.
Disable with `--no_resume`. Job 2 is identical at the per-config level
(`run_ft_grid.sh` skips configs whose `results.json` exists; raise `--time` if
needed, then resubmit).

## After the runs
`python training/summarize_finetune.py` aggregates `results/ft/*`. Apply the
pre-registered decision rules in `README_FINETUNE.md` (frozen sanity gate
0.90±0.02; adopt FT only if dev-ensemble ≥0.92; headline by OOF AUC).
Compare the official-test FT numbers against the frozen single bar (macro 0.774
/ AUC 0.864) and the ensemble (macro 0.833).
