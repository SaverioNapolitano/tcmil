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
- job 3 → `cluster/finalists_configs.txt` (**edit after job 2**: top-2 grid configs by 5-seed dev-ensemble AUC + frozen)
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

## After the runs
`python training/summarize_finetune.py` aggregates `results/ft/*`. Apply the
pre-registered decision rules in `README_FINETUNE.md` (frozen sanity gate
0.90±0.02; adopt FT only if dev-ensemble ≥0.92; headline by OOF AUC).
Compare the official-test FT numbers against the frozen single bar (macro 0.774
/ AUC 0.864) and the ensemble (macro 0.833).
