# Reports — DAIC-WOZ Text-Only Depression Detection

Authoritative, leak-free documentation. The per-version model-evolution cards
and the pooled-189 leaderboard that used to live here were **deleted**: their
numbers came from a leaky pooled-CV protocol and the version ranking was a
protocol artifact (under the rigorous protocol no distinct SS-DAMIL-R variant
beats the V9 architecture, and all trail TC-MIL). The architecture ladder and
its provenance are preserved in the ablation study below.

## Documents

- **[../../README_TCMIL.md](../../README_TCMIL.md)** — the model: design,
  full-189 headline results, official split + cross-validation, bias control.
- **[../../results/ABLATION-STUDY.md](../../results/ABLATION-STUDY.md)** —
  architecture ladder (dialogue-mean → flat MIL → DAMIL-R → SS-DAMIL-R
  v9/v25/v26/v29 → TC-MIL) under the leak-free protocol, per-seed mean ± std,
  and the note on why the old version leaderboard was a protocol artifact.
- **[../../results/full189/SEED_STABILITY.md](../../results/full189/SEED_STABILITY.md)**
  — single-model vs ensemble, 30-seed mean ± std, significance vs the baseline.
- **[../../results/ablation_stats.md](../../results/ablation_stats.md)** —
  statistical tests (Wilcoxon, Mann-Whitney U, paired/independent t-tests),
  TC-MIL vs each legacy baseline.
- **[ALTERNATIVE-APPROACHES.md](ALTERNATIVE-APPROACHES.md)** — literature
  review for **baseline selection**: which published DAIC-WOZ results are valid
  to compare against (primary baseline **Milintsevich et al. 2023**) and which
  are bias-suspect or unverified.
- **[INTERPRETABILITY.md](INTERPRETABILITY.md)** — interpretability
  methodology (attention faithfulness, PHQ-8 read-out, bias probes).

## Leak-free headline (full 189 transcripts, official split)

| Architecture | official macro-F1 (single, 30-seed) | official AUC |
| --- | --- | --- |
| DAMIL-R | 0.616 ± 0.121 | 0.753 ± 0.065 |
| SS-DAMIL-R v9 (best legacy) | 0.657 ± 0.080 | 0.758 ± 0.051 |
| **TC-MIL (chunks + GRU)** | **0.751 ± 0.032** | **0.856 ± 0.011** |

Full ladder, CV columns, and significance tests: `results/ABLATION-STUDY.md`.
