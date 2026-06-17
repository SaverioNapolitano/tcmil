# Fine-tuning iteration 2

Prepared while job-2 (stage-1 grid) was still running. Grounded in the **job-1
(DAPT) results only** — grid-driven refinements wait for job-2.

## What job-1 told us

`checkpoints/dapt_bge_large/dapt.log` (3 epochs, train loss only):

| epoch | mean MLM loss |
|------:|--------------:|
| 1 | 6.16 |
| 2 | 3.87 |
| 3 | 3.19 |

Two problems:

1. **Undertrained.** Loss was still falling ~-0.68/epoch at the last epoch —
   nowhere near a plateau. A good MLM on this text should sit well below 3.2.
   The whole run took ~2.5 min, so more epochs are nearly free.
2. **No validation signal.** Only train loss is logged, so "train longer"
   cannot be justified without risking overfit on a tiny 107-interview corpus.

**Verdict: a second DAPT iteration would help — but only if DAPT earns its place
in the grid.** Whether domain-adaptive pretraining helps *downstream* at all is
decided by job-2 (the two `dapt_*` grid configs), not by the MLM loss.

## Decision rule (apply after job-2)

Run `python src/statistics/summarize_finetune.py --root results/ft`, then:

- **DAPT arm competitive** — a `dapt_*` config lands within seed noise of (or
  above) the best non-DAPT grid config on 5-seed dev AUC → **run DAPT v2**
  (`ft_job1b_dapt_v2.sh`), then add a `dapt_v2_*` line to the stage-2 finalists
  pointing at `checkpoints/dapt_bge_large_v2`.
- **DAPT arm clearly worse** — drop the DAPT arm entirely; iteration 2 becomes a
  pure HP-refinement round around the stage-1 winners (no DAPT). DAPT v2 not
  worth the compute.

## DAPT v2 — what changed vs v1

`src/training/dapt_mlm_v2.py` (standalone; same leakage guarantee as v1 —
encoder sees the official **train split only**):

- **More epochs** (15) with early stopping.
- **Held-out MLM validation**: an *interview-level* 10% hold-out carved from the
  train split (no chunk of a val interview is in MLM-train; no dev/test text) →
  per-epoch val MLM loss, leakage-free.
- **Best-by-val checkpoint** + **early stop** (patience 3) → stop where val loss
  bottoms, not at a fixed epoch count.
- Fixed `eval_seed` so val masking is identical every epoch → comparable val
  losses across epochs.

Run:

```bash
sbatch cluster/iteration_2/ft_job1b_dapt_v2.sh   # -> checkpoints/dapt_bge_large_v2
```

## Not done yet (needs job-2)

- Stage-1 HP refinement (narrow lr / lora_r / unfreeze_last_k around the grid
  winners). Ranges depend on `summarize_finetune.py` output — fill in once job-2
  finishes.
