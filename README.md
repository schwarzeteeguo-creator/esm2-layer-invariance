# Depth Matters but the Final Layer Suffices: Leakage-Free Evaluation of Layer Choice in Protein Language Models for Mutation Effect Prediction


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, per-dataset results, and manuscript for our paper in revision at
**Bioinformatics** (BIOINF-2026-2348). This tag of the repository corresponds to
the revised resubmission; the earlier history (directories `code/`, `analysis/`,
and the root `results/` and `manuscript/` files dated before September 2026)
corresponds to the originally submitted version and is retained unchanged as the
record of what was withdrawn and why.

## What v3 (second revision) adds

- **Retitled and reframed**: the core claim is now final-layer sufficiency, not
  layer invariance (best-versus-final within 0.01 rho across 20 leakage-free
  TOST comparisons: 8 equivalent, 12 inconclusive, 0 beyond the margin; 7
  further combinations have the final layer as the best layer).
- **Unified-protocol pooling under the random split** (`results/r2/a2_a3_pooling.json`,
  `analysis/analysis_r2.py`): small significant fusion advantages appear under
  the random split (+0.009 to +0.050) and vanish, sometimes reversing, under
  leakage-free evaluation.
- **Published ProteinGym anchors recomputed on the same assays**
  (`results/r2/r2_anchors_zeroshot.json`, `r2_anchors_supervised.json`,
  `esm_embedding/pg_anchor_scores.py`): official ESM-2 650M 0.472 vs our
  validated lens 0.438; Kermut 0.605 / ProteinNPT 0.566 under the official
  modulo fold; the published one-hot baseline collapses from 0.570 (random)
  to -0.003 (modulo).
- **CKA at 200 positions + debiased estimator** (`results/r2/cka200_*`,
  updated `esm_embedding/cka.py`): SaProt L0-L32 0.78, ESM-2 0.41.
- **Single-substitution-only sensitivity analysis**
  (`results/r2/probing_v4_single_*`, `analysis/analysis_single_site.py`,
  `--single_site_only` in `probing_v4.py`): depth trend and structure-token
  benefit replicate; final layer within 0.01 of best for SaProt, within 0.02
  for ESM-2.
- **Envision-style supervised baseline** (`esm_embedding/envision_baseline.py`,
  `results/r2/envision_per_dataset.json`): 0.634 random vs 0.410 modulo /
  0.302 contiguous under identical folds.
- **Assay composition audit** (`esm_embedding/multisite_stats.py`,
  `results/r2/r2_multisite_lmdb.json`): 84.6% of variants are multi-site,
  contributing 95.8% of analyzed rows; caps affect 11 (10k variants) and 26
  (5k rows) of 63 datasets.

## What the (first) revision shows

- **Random-split layer invariance is largely position memorization.** A Ridge
  regressor given only random vectors indexed by mutation position reproduces the
  published random-split probing level (rho = 0.600) and collapses to 0.10-0.14
  under leakage-free splits (paired drop -0.46, Wilcoxon p = 5.2e-12).
- **Under leakage-free splits** (ProteinGym-official modulo and contiguous, plus
  position-grouped), mutation-specific embeddings reach rho up to 0.53, the best
  layers are consistently 30-32, and best-versus-final differences never exceed
  0.009 (TOST margin 0.01: 7/19 equivalent, 12 indeterminate, none beyond margin).
- **Structure tokens help mainly under leakage-free evaluation** (+0.06 to +0.12,
  all p <= 1e-4), which explains why they appeared useless in random-split
  comparisons.
- **Withdrawn results** (disclosed in the revision): the ESM-2 pooling asymmetry
  (0.386 vs 0.639, a cross-protocol artifact; 0.463 vs 0.481 under the unified
  pipeline), the non-monotonic fp16 logit-lens curve (validated fp32 curves are
  monotonic with the final layer best: ESM-2 0.443, SaProt 0.437), and the
  U-shaped similarity recovery (recomputed curves are monotone; linear CKA
  between SaProt layers 0 and 32 is 0.81 while cosine is -0.009).

## Repository structure

```
esm_embedding/                  revision pipeline (python -m esm_embedding.*)
  probing_v4.py                   three-arm x four-split probing + position control
  pg_splits.py                    official split schemes
  benchmark_v3.py                 pooling with per-fold-trained attention
  logit_lens_v2.py                fp32 lens with per-dataset validation gate
  cka.py                          cosine + linear CKA
  stats_tost.py                   TOST / bootstrap CI / d_z
  aggregate_results.py            rebuilds the master table
pipelines/                      server orchestration + 3Di gap-fix utilities
results/rev/                    revision results
  server_results/                 probing x3 suites, pooling x2, lens x2, CKA x2
  probing_v4_gcv/                 random-position control (GCV protocol)
  3di_server/                     Foldseek 3Di tokens (62 datasets)
  old_ll_fp16_repro/              reproduction record of the withdrawn fp16 curves
  MAIN_TABLE.md                   master results table (all numbers with n)
manuscript/
  manuscript_revised_v2.*         red-marked revised manuscript
  supplementary_v2.pdf            supplementary tables S1-S5
code/, analysis/, results/*.json, manuscript/manuscript.tex
                                submitted version (unchanged, for the record)
```

See `results/rev/DEPOSIT_README.md` for environment details, seeds, JSON schemas,
and reproduction commands.

## Data

ProteinGym substitution benchmark (Notin et al., 2024). Structure tokens from
AlphaFold Protein Structure Database alignments via Foldseek (62 of 63 datasets;
two C-terminally truncated and gap-padded; influenza nucleoprotein
structure-masked).

## License

Code: MIT. Results and 3Di token files: CC-BY-4.0.

## Archived version

A snapshot of this repository is archived on Zenodo with a DOI (see the release
sidebar). The DOI is cited in the revised manuscript's Data Availability section.
