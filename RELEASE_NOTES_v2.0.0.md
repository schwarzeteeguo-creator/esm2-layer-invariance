# Release v2.0.0 — Revision of BIOINF-2026-2348

Revision snapshot accompanying the resubmission to Bioinformatics.

## What changed since the submitted version (v1)

- **New unified protocol.** All supervised experiments re-run under one pipeline:
  three feature arms (masked_wt / unmasked_wt / mutant) x four splits (random,
  ProteinGym-official modulo and contiguous, position-grouped), Ridge with
  generalized cross-validation, standardization inside training folds, all 63
  ProteinGym substitution datasets.
- **Position-leakage control.** A random-position lookup table reproduces the
  random-split probing level (rho = 0.600) and collapses to 0.10-0.14 under
  leakage-free splits; the paper's random-split layer invariance is reinterpreted
  as largely position memorization.
- **Withdrawn results, disclosed.** ESM-2 pooling asymmetry (0.386 vs 0.639,
  cross-protocol artifact), non-monotonic fp16 logit-lens curve (fp32 validated
  curves are monotonic, final layer best: 0.443 / 0.437), U-shaped similarity
  recovery (recomputed curves are monotone).
- **New analyses.** Linear CKA (rotation-with-preservation), TOST equivalence
  tests (margin 0.01), real Foldseek 3Di structure tokens (62/63 datasets) showing
  a conditional structure-token benefit (+0.06 to +0.12, p <= 1e-4, leakage-free
  only).
- **Title updated** to "Conditional Layer Invariance in Protein Language Models:
  Position Leakage and Readout Choice Determine the Apparent Benefit of
  Multi-Scale Embeddings".

## Contents

- `esm_embedding/` — revision pipeline; `results/rev/` — complete per-dataset
  results; `manuscript/manuscript_revised_v2.*` — red-marked revised manuscript;
  `manuscript/supplementary_v2.pdf` — supplementary tables S1-S5.
- Submitted-version directories retained unchanged as the withdrawal record.

Code MIT; results and 3Di token files CC-BY-4.0.
