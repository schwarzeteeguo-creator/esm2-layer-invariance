# Conditional Layer Invariance in Protein Language Models: Position Leakage and Readout Choice Determine the Apparent Benefit of Multi-Scale Embeddings

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, per-dataset results, and manuscript for our paper in revision at
**Bioinformatics** (BIOINF-2026-2348). This tag of the repository corresponds to
the revised resubmission; the earlier history (directories `code/`, `analysis/`,
and the root `results/` and `manuscript/` files dated before September 2026)
corresponds to the originally submitted version and is retained unchanged as the
record of what was withdrawn and why.

## What the revision shows

- **Random-split layer invariance is largely position memorization.** A Ridge
  regressor given only random vectors indexed by mutation position reproduces the
  published random-split probing level (rho = 0.600) and collapses to 0.10-0.14
  under leakage-free splits (paired drop -0.46, Wilcoxon p = 5.2e-12).
- **Under leakage-free splits** (ProteinGym-official modulo and contiguous, plus
  position-grouped), mutation-specific embeddings reach rho up to 0.53, the best
  layers are consistently 30-32, and best-versus-final differences never exceed
  0.009 (TOST margin 0.01: 7/19 equivalent, 12 indeterminate, none beyond margin).
- **Structure tokens help only under leakage-free evaluation** (+0.06 to +0.12,
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
